"""Smart 1-to-1 Disambiguation & Submission Finalizer.

Resolves target collisions across France, US, and India:
- In France, eliminates ~140,000 cross-street false claims on generic business names
  and awards contested targets to the entity with genuine matching street address.
- In US and India, resolves all contested targets using street-token weighted scores.
- Maintains strict 1-to-1 target uniqueness (no target claimed by multiple entities).
- Writes matching_results.tsv matching test_source1.tsv entity order with pure UNIX LF.
"""
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PARTS = ROOT / "dataset" / "test" / "partitions"
TEST = ROOT / "dataset" / "test"
OUT = ROOT / "output"
SCR = ROOT / "scratch" / "v4"

ADDR_STOP = {
    "near", "opp", "opposite", "behind", "beside", "adj", "adjacent",
    "at", "post", "po", "dist", "district", "taluk", "tehsil", "road",
    "rd", "street", "st", "lane", "ln", "avenue", "ave", "highway",
    "cross", "main", "phase", "sector", "sec", "block", "floor",
    "room", "flat", "shop", "gala", "bldg", "building", "house",
    "complex", "nagar", "colony", "enclave", "vihar", "layout",
    "city", "state", "pin", "code", "zip", "india", "us", "usa", "france",
    "rue", "chemin", "quai", "allee", "route", "cedex", "bp", "impasse", "imp",
    "du", "de", "la", "le", "les", "des", "et", "chez", "sur", "au", "aux", "d", "l",
    "cours", "crs", "boulevard", "bd", "blvd", "passage", "residence", "batiment", "bat", "apt",
    "suite", "ste", "dr", "drive", "ct", "court", "pl", "place", "sq", "square",
    "nouvelle", "aquitaine", "bordeaux", "ile", "paris", "occitanie", "toulouse", "lyon",
    "provence", "alpes", "cote", "dazur", "marseille", "auvergne", "rhone", "alpes",
    "hauts", "normandie", "bretagne", "grand", "est", "strasbourg", "lille", "nantes"
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def core_toks(a: str) -> set:
    if not a:
        return set()
    tokens = re.findall(r"[a-z0-9]+", a.lower())
    return set(t for t in tokens if t not in ADDR_STOP and not t.isdigit() and len(t) > 2)


def process_france():
    log("=== Processing FRANCE with Street-Level Disambiguation ===")
    t0 = time.time()
    s1_addrs = {}
    s1_order = []
    with open(PARTS / "france_s1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip().split("\t")
            s1_addrs[p[0]] = p[2]
            s1_order.append(p[0])

    other_addrs = {}
    with open(PARTS / "france_other.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip().split("\t")
            other_addrs[p[0]] = p[2]
    log(f"France records loaded in {time.time()-t0:.1f}s")

    best = {}  # target -> (score, s1)
    n_claims = 0
    dropped_zero_overlap = 0

    with open(SCR / "claims_france.tsv", "r", encoding="utf-8") as f:
        for line in f:
            s1, tid, pr = line.rstrip().split("\t")
            pr = float(pr)
            n_claims += 1
            t1 = core_toks(s1_addrs.get(s1, ""))
            t2 = core_toks(other_addrs.get(tid, ""))
            if t1 and t2 and not (t1 & t2):
                dropped_zero_overlap += 1
                continue
            overlap_score = len(t1 & t2) / max(len(t1 | t2), 1) if (t1 and t2) else 0.5
            score = pr * (0.50 + 0.50 * overlap_score)
            cur = best.get(tid)
            if cur is None or score > cur[0] or (score == cur[0] and s1 < cur[1]):
                best[tid] = (score, s1)

    log(f"France: {n_claims:,} claims -> dropped {dropped_zero_overlap:,} cross-street false claims "
        f"({dropped_zero_overlap/n_claims*100:.1f}%) -> {len(best):,} targets assigned")

    # Group by S1
    s1_matches = defaultdict(list)
    for tid, (sc, s1) in best.items():
        s1_matches[s1].append(tid)

    out_path = OUT / "matching_results_france.tsv"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in s1_order:
            mlist = sorted(s1_matches.get(s1, []))
            f.write(s1 + "\t" + ",".join(mlist) + "\n")

    singles = sum(1 for s in s1_order if not s1_matches.get(s))
    log(f"France saved -> {out_path.name}: {len(s1_order):,} entities, "
        f"{singles:,} singletons ({singles/len(s1_order)*100:.2f}%)")
    return s1_matches


def process_country_fast(ctry: str):
    log(f"=== Processing {ctry.upper()} Fast Targeted Disambiguation ===")
    t0 = time.time()
    # Pass 1: count target claims
    target_counts = Counter()
    with open(SCR / f"claims_{ctry}.tsv", "r", encoding="utf-8") as f:
        for line in f:
            s1, tid, pr = line.rstrip().split("\t")
            target_counts[tid] += 1

    contested_targets = {t for t, c in target_counts.items() if c > 1}
    log(f"[{ctry}] Total targets: {len(target_counts):,}, Contested targets: {len(contested_targets):,}")

    # Pass 2: collect claims
    contested_claims = defaultdict(list)
    uncontested_best = {}
    contested_s1 = set()

    with open(SCR / f"claims_{ctry}.tsv", "r", encoding="utf-8") as f:
        for line in f:
            s1, tid, pr = line.rstrip().split("\t")
            pr = float(pr)
            if tid in contested_targets:
                contested_claims[tid].append((s1, pr))
                contested_s1.add(s1)
            else:
                uncontested_best[tid] = s1

    log(f"[{ctry}] Loading addresses for {len(contested_s1):,} contested S1 and "
        f"{len(contested_targets):,} contested targets...")
    s1_addrs = {}
    s1_order = []
    with open(PARTS / f"{ctry}_s1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip().split("\t")
            s1_order.append(p[0])
            if p[0] in contested_s1:
                s1_addrs[p[0]] = p[2]

    other_addrs = {}
    with open(PARTS / f"{ctry}_other.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip().split("\t")
            if p[0] in contested_targets:
                other_addrs[p[0]] = p[2]
                if len(other_addrs) >= len(contested_targets):
                    break

    # Resolve contested targets
    resolved = dict(uncontested_best)
    dropped_cross = 0

    for tid, claims in contested_claims.items():
        t2 = core_toks(other_addrs.get(tid, ""))
        scored = []
        for s1, pr in claims:
            t1 = core_toks(s1_addrs.get(s1, ""))
            if t1 and t2 and not (t1 & t2):
                dropped_cross += 1
                continue
            overlap_score = len(t1 & t2) / max(len(t1 | t2), 1) if (t1 and t2) else 0.5
            score = pr * (0.50 + 0.50 * overlap_score)
            scored.append((score, pr, s1))

        if scored:
            scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
            resolved[tid] = scored[0][2]
        else:
            # Fallback to highest prob
            claims.sort(key=lambda x: -x[1])
            resolved[tid] = claims[0][0]

    log(f"[{ctry}] Resolved {len(resolved):,} targets in {time.time()-t0:.1f}s "
        f"(dropped {dropped_cross:,} cross-street candidates)")

    # Group by S1
    s1_matches = defaultdict(list)
    for tid, s1 in resolved.items():
        s1_matches[s1].append(tid)

    out_path = OUT / f"matching_results_{ctry}.tsv"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in s1_order:
            mlist = sorted(s1_matches.get(s1, []))
            f.write(s1 + "\t" + ",".join(mlist) + "\n")

    singles = sum(1 for s in s1_order if not s1_matches.get(s))
    log(f"[{ctry}] saved -> {out_path.name}: {len(s1_order):,} entities, "
        f"{singles:,} singletons ({singles/len(s1_order)*100:.2f}%)")
    return s1_matches


def assemble_final_submission():
    log("=== Assembling Final matching_results.tsv ===")
    t0 = time.time()
    
    # Load all country match mappings
    all_matches = {}
    for ctry in ("france", "us", "india"):
        p = OUT / f"matching_results_{ctry}.tsv"
        log(f"Reading {p.name}...")
        with open(p, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                s1, _, rest = line.rstrip("\r\n").partition("\t")
                all_matches[s1] = rest

    # Read test_source1.tsv to guarantee 100% exact entity order
    final_path = OUT / "matching_results.tsv"
    total_written = 0
    total_singletons = 0
    total_matches = 0

    with open(TEST / "test_source1.tsv", "r", encoding="utf-8") as f_in, \
         open(final_path, "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        next(f_in)
        for line in f_in:
            eid = line.rstrip("\r\n").split("\t")[0]
            mlist = all_matches.get(eid, "")
            f_out.write(eid + "\t" + mlist + "\n")
            total_written += 1
            if not mlist:
                total_singletons += 1
            else:
                total_matches += len(mlist.split(","))

    log(f"Final matching_results.tsv written in {time.time()-t0:.1f}s:")
    log(f"  Total S1 entities: {total_written:,}")
    log(f"  Total Singletons:  {total_singletons:,} ({total_singletons/total_written*100:.2f}%)")
    log(f"  Total Matches:     {total_matches:,}")


def main():
    log("Starting Smart Disambiguation & Submission Finalization Pipeline")
    process_france()
    process_country_fast("us")
    process_country_fast("india")
    assemble_final_submission()
    log("Pipeline complete! Proceed to validation.")


if __name__ == "__main__":
    main()
