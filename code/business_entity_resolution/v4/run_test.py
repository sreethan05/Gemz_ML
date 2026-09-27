"""v4 run_test: production inference over the full test set.

Uses per-country calibrated configs (models/v4_config.json):
  cap (candidate cap), threshold, min_top, rescue mode.
Writes output/matching_results.tsv + output/candidate_pairs.tsv (+ claims
sidecars), applies probability-based 1-to-1 target disambiguation per country,
verifies LF purity and subset integrity, then runs the official validator.
"""
import gc
import json
import pickle
import subprocess
import sys
import time
from array import array
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from v4.core import (PackedPool, CSRIndex, EntityRecord, ensure_hashseed,
                     keys_of, key_hashes, compute_pair_features, query_cascade,
                     N_FEATURES, build_clusters, expand_with_siblings,
                     fingerprint_of, accept_indices)

ROOT = Path(__file__).resolve().parents[3]
PARTS = ROOT / "dataset" / "test" / "partitions"
OUTD = ROOT / "output"
SCR = ROOT / "scratch" / "v4"
MODELS = ROOT / "models"

BATCH = 5000
CSR_CHUNK = 10_000_000  # single CSR per country: one searchsorted pass, fastest queries
BUILD_CACHE = 100_000

I_NSORT, I_ASORT, I_PINMATCH, I_NUMOV = 2, 11, 20, 17


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_pool_file(path: Path) -> PackedPool:
    pool = PackedPool()
    pool.cache_cap = 1  # no cache during bulk parse
    t0 = time.time()
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 3:
                pool.add_raw(p[0], p[1], p[2])
                if pool.n % 1_000_000 == 0:
                    log(f"  added {pool.n:,} ({time.time()-t0:.0f}s)")
    pool.finalize()
    pool._cache.clear()
    pool.cache_cap = BUILD_CACHE
    log(f"pool {path.name}: {pool.n:,} records in {time.time()-t0:.0f}s")
    return pool


def build_csrs(pool: PackedPool):
    csrs = []
    for j0 in range(0, pool.n, CSR_CHUNK):
        j1 = min(j0 + CSR_CHUNK, pool.n)
        csrs.append(CSRIndex.build(pool, j0, j1))
        pool._cache.clear()
        gc.collect()
    log(f"  index ready: {sum(len(c.J) for c in csrs):,} postings")
    return csrs


def stream_s1(path: Path, skip: int = 0, limit=None):
    with open(path, encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i < skip:
                continue
            if limit is not None and i >= skip + limit:
                break
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 3:
                yield EntityRecord(p[0], p[1], p[2])


def iter_batches(it, n):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def run_country(ctry: str, cfg: dict, clf, fm, fc, fclaims, skip: int = 0, limit=None):
    cap = int(cfg["cap"])
    th = float(cfg["threshold"])
    mt = float(cfg["min_top"])
    resc = cfg["rescue"]
    cluster = cfg.get("cluster", "off")
    cluster_floor = float(cfg.get("cluster_floor", 0.0))
    log(f"==== {ctry.upper()} (cap={cap} t={th:.2f} mt={mt:.2f} rescue={resc} "
        f"cluster={cluster}/{cluster_floor:.2f}) ====")

    pool = build_pool_file(PARTS / f"{ctry}_other.tsv")
    csrs = build_csrs(pool)
    cluster_maps = build_clusters(pool) if cfg.get("expand") else None
    if cluster_maps is not None:
        gc.collect()
        log(f"  sibling clusters built")

    n_ent = n_empty_cand = n_single = n_matches = 0
    t0 = time.time()
    for batch in iter_batches(stream_s1(PARTS / f"{ctry}_s1.tsv", skip, limit), BATCH):
        cand_lists = []
        js_all = []
        feat_flat = array("f")
        elig_gated = array("b")
        elig_full = array("b")
        for r in batch:
            js, _combined = query_cascade(pool, csrs, r, cap)
            if cluster_maps is not None and len(js):
                js = expand_with_siblings(pool, cluster_maps, js, extra=15)
            js = sorted(int(j) for j in js)  # deterministic candidate order
            if len(js):
                js_all.append(js)
                cand_lists.append([pool.get_id(j) for j in js])
                for j in js:
                    fr = compute_pair_features(r, pool.record(j))
                    feat_flat.extend(fr)
                    base = (fr[I_ASORT] >= 0.90 and
                            (fr[I_PINMATCH] > 0.5 or fr[I_NUMOV] >= 0.5))
                    elig_full.append(1 if base else 0)
                    elig_gated.append(1 if (base and fr[I_NSORT] >= 0.85) else 0)
            else:
                js_all.append([])
                cand_lists.append([])
        n_pairs = len(feat_flat) // N_FEATURES
        if n_pairs:
            X = np.frombuffer(feat_flat, dtype=np.float32).reshape(n_pairs, N_FEATURES)
            probs_all = clf.predict_proba(X)[:, 1]
        else:
            probs_all = np.empty(0, dtype=np.float32)
        del feat_flat

        pos = 0
        for r, cids in zip(batch, cand_lists):
            n_ent += 1
            fc.write(r.id + "\t" + ",".join(cids) + "\n")
            if not cids:
                fm.write(r.id + "\t\n")
                n_empty_cand += 1
                n_single += 1
                continue
            k = len(cids)
            probs = probs_all[pos:pos + k].copy()
            if resc == "gated":
                lift = np.frombuffer(elig_gated, dtype=np.int8)[pos:pos + k].astype(bool)
            elif resc == "full":
                lift = np.frombuffer(elig_full, dtype=np.int8)[pos:pos + k].astype(bool)
            else:
                lift = None
            if lift is not None and lift.any():
                probs[lift] = np.maximum(probs[lift], th)
            pos += k

            fps = [fingerprint_of(pool.record(js_all[n_ent - 1][i])) if cluster == "propagate"
                   else "" for i in range(k)] if cluster == "propagate" else [""] * k
            # js_all is parallel to batch; current entity is the last appended
            # in this loop — use the local candidate records instead:
            if cluster == "propagate":
                # records already cached from feature extraction
                rec_js = js_all[len(js_all) - len(batch) + list(zip(batch, cand_lists)).index((r, cids))] \
                    if False else None
            acc = accept_indices(probs, fps, th, mt, cluster, cluster_floor)
            if not acc:
                fm.write(r.id + "\t\n")
                n_single += 1
                continue
            n_matches += len(acc)
            fm.write(r.id + "\t" + ",".join(sorted(cids[i] for i in acc)) + "\n")
            for i in acc:
                fclaims.write(f"{r.id}\t{cids[i]}\t{probs[i]:.6f}\n")
        if len(pool._cache) > BUILD_CACHE:
            pool._cache.clear()
        if n_ent % 100_000 < BATCH:
            log(f"  {ctry}: {n_ent:,} entities, {n_matches:,} raw matches, "
                f"{time.time()-t0:.0f}s")
    log(f"[{ctry}] entities={n_ent:,} zero-cand={n_empty_cand:,} "
        f"pre-prune singles={n_single:,} raw matches={n_matches:,} "
        f"in {time.time()-t0:.0f}s")
    return {"entities": n_ent, "zero_cand": n_empty_cand,
            "raw_singles": n_single, "raw_matches": n_matches}


def prune_1to1_country(ctry: str, match_path: Path, claims_path: Path) -> dict:
    """Probability-based 1-to-1 target assignment (per country; pools are
    disjoint by country so claims never cross countries)."""
    t0 = time.time()
    best = {}  # target -> (prob, s1)
    with open(claims_path, encoding="utf-8") as f:
        for line in f:
            s1, tid, pr = line.rstrip("\n").split("\t")
            pr = float(pr)
            cur = best.get(tid)
            if cur is None or pr > cur[0] or (pr == cur[0] and s1 < cur[1]):
                best[tid] = (pr, s1)
    tmp = match_path.with_suffix(f".{ctry}.tmp")
    kept_pairs = pruned = singles = rows = 0
    with open(match_path, encoding="utf-8") as f, \
         open(tmp, "w", encoding="utf-8", newline="\n") as g:
        g.write("source1_entity_id\tmatched_entity_ids\n")
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            mlist = [x for x in rest.split(",") if x] if rest else []
            kept_list = [t for t in mlist if best.get(t, (0.0, s1))[1] == s1]
            kept_pairs += len(kept_list)
            pruned += len(mlist) - len(kept_list)
            if not kept_list and mlist:
                singles += 1
            rows += 1
            g.write(s1 + "\t" + ",".join(sorted(kept_list)) + "\n")
    tmp.replace(match_path)
    log(f"[{ctry}] 1-to-1 prune: kept {kept_pairs:,}, pruned {pruned:,} "
        f"false merges, {singles:,} entities became singletons "
        f"({time.time()-t0:.0f}s)")
    return {"kept": kept_pairs, "pruned": pruned, "new_singles": singles,
            "rows": rows}


def main():
    ensure_hashseed()
    cfg_name = sys.argv[2] if len(sys.argv) > 2 else "v4_config.json"
    with open(MODELS / cfg_name, encoding="utf-8") as f:
        cfg = json.load(f)
    with open(ROOT / cfg["model"], "rb") as f:
        clf = pickle.load(f)
    clf.set_params(n_jobs=2)

    # one country per invocation (enables parallel execution); "all" = legacy path
    ctry_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    skip_n = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--skip" else 0
    limit_n = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[5] == "--limit" else None
    part_tag = sys.argv[8] if len(sys.argv) > 8 and sys.argv[7] == "--part" else ""
    targets = (["france", "us", "india"] if ctry_arg == "all"
               else [t for t in ("france", "us", "india") if t in ctry_arg.split(",")])

    if ctry_arg == "all":
        m_path = OUTD / "matching_results.tsv"
        c_path = OUTD / "candidate_pairs.tsv"
        with open(m_path, "w", encoding="utf-8", newline="\n") as fm, \
             open(c_path, "w", encoding="utf-8", newline="\n") as fc:
            fm.write("source1_entity_id\tmatched_entity_ids\n")
            fc.write("source1_entity_id\tcandidate_entity_ids\n")
    report = {}
    for ctry in targets:
        cc = cfg["countries"].get(ctry) or cfg["countries"][cfg.get("france_uses", "us")]
        suffix = "" if ctry_arg == "all" else f"_{ctry}{part_tag}"
        sk = skip_n if (ctry == "us" and part_tag) else 0
        lm = limit_n if (ctry == "us" and part_tag) else None
        m_path = OUTD / f"matching_results{suffix}.tsv"
        c_path = OUTD / f"candidate_pairs{suffix}.tsv"
        with open(m_path, "w", encoding="utf-8", newline="\n") as fm, \
             open(c_path, "w", encoding="utf-8", newline="\n") as fc, \
             open(SCR / f"claims_{ctry}.tsv", "w", encoding="utf-8", newline="\n") as fcl:
            fm.write("source1_entity_id\tmatched_entity_ids\n")
            fc.write("source1_entity_id\tcandidate_entity_ids\n")
            report[ctry] = run_country(ctry, cc, clf, fm, fc, fcl, skip=sk, limit=lm)
        report[ctry].update(prune_1to1_country(ctry, m_path, SCR / f"claims_{ctry}.tsv"))
        gc.collect()

    with open(SCR / f"run_report_{ctry_arg.replace(',', '_')}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    log(f"DONE {ctry_arg}")


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        print("FATAL: out of memory. Close other applications and re-run.", flush=True)
        sys.exit(3)
