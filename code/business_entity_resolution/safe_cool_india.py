"""
Safe, Thermal-Protected, Lean-Memory India Partition Inference Engine
Team Gemz - Amazon ML Challenge 2026

Loads ONLY what is necessary:
  - Remaining 400,028 India S1 entities (no re-loading completed France/US).
  - Lightweight raw India Other tuples (takes ~1.7 GB instead of 5 GB).
  - On-demand feature extraction for candidates only.
  - Windows BELOW_NORMAL priority + 2 CPU threads + 3.0s cooldown between batches.
"""

import sys
import os
import gc
import csv
import re
import json
import time
import pickle
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

import numpy as np
import lightgbm as lgb
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz import fuzz

# Strict thread limits: 2 threads only to keep CPU cool and prevent overheating
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

LEGAL_TOKENS = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation",
    "co", "company", "pvt", "private", "llc", "llp", "gmbh", "sa",
    "sarl", "bv", "nv", "plc", "holdings", "group", "enterprises",
    "enterprise", "services", "solutions", "technologies", "tech",
    "international", "consulting", "associates", "sons", "trading",
    "industries", "traders", "agency", "agencies", "stores", "store"
}

ADDR_STOP = {
    "near", "opp", "opposite", "behind", "beside", "adj", "adjacent",
    "at", "post", "po", "dist", "district", "taluk", "tehsil", "road",
    "rd", "street", "st", "lane", "ln", "avenue", "ave", "highway",
    "cross", "main", "phase", "sector", "sec", "block", "floor",
    "room", "flat", "shop", "gala", "bldg", "building", "house",
    "complex", "nagar", "colony", "enclave", "vihar", "layout",
    "city", "state", "pin", "code", "zip", "india"
}

ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "hwy": "highway", "dr": "drive", "ct": "court",
    "pl": "place", "sq": "square", "ter": "terrace", "apt": "apartment",
    "ste": "suite", "bldg": "building", "no": "number", "num": "number",
    "mgr": "marg", "sect": "sector", "sec": "sector", "ph": "phase",
    "gnd": "ground", "flt": "flat", "hse": "house", "soc": "society",
    "xing": "crossing", "chowk": "chowk", "br": "branch"
}

_NON_ALNUM = re.compile(r"[^\w\s]+", re.UNICODE)
_NUM_RE = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("&", " and ")
    s = s.replace(".", " ").replace("/", " ").replace("-", " ")
    s = _NON_ALNUM.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


class IndiaParsed:
    __slots__ = (
        'id', 'name_norm', 'addr_norm', 'core_toks', 'addr_toks',
        'numbers', 'pins', 'first_tok', 'is_s2'
    )

    def __init__(self, eid: str, name: str, address: str):
        name_norm = clean_text(name)
        addr_raw = clean_text(address)
        addr_toks = [ADDR_ABBREV.get(t, t) for t in addr_raw.split()]
        name_toks = name_norm.split()

        self.id = eid
        self.name_norm = name_norm
        self.addr_norm = " ".join(addr_toks)
        self.core_toks = tuple(t for t in name_toks if t not in LEGAL_TOKENS and len(t) >= 3)
        self.addr_toks = frozenset(t for t in addr_toks if t not in ADDR_STOP and not t.isdigit() and len(t) >= 6)
        nums = tuple(sorted(set(n.lstrip("0") or "0" for t in addr_toks for n in _NUM_RE.findall(t) if len(n) >= 1)))
        self.numbers = nums
        self.pins = frozenset(n for n in nums if len(n) == 6)
        self.first_tok = name_toks[0] if name_toks else ""
        self.is_s2 = eid.startswith("S2-")


def extract_blocking_keys_raw(name: str, address: str) -> set:
    name_norm = clean_text(name)
    addr_raw = clean_text(address)
    addr_toks = [ADDR_ABBREV.get(t, t) for t in addr_raw.split()]
    name_toks = name_norm.split()

    core_toks = [t for t in name_toks if t not in LEGAL_TOKENS and len(t) >= 3]
    addr_toks_f = [t for t in addr_toks if t not in ADDR_STOP and not t.isdigit() and len(t) >= 6]
    nums = sorted(set(n.lstrip("0") or "0" for t in addr_toks for n in _NUM_RE.findall(t) if len(n) >= 1))
    pins = [n for n in nums if len(n) == 6]
    f2 = name_norm[:2] if len(name_norm) >= 2 else name_norm

    keys = set()
    for p in pins:
        keys.add(("p", p))
        if f2:
            keys.add(("pf", p, f2))

    for t in core_toks:
        if len(t) >= 4:
            keys.add(("n", t))

    if len(core_toks) >= 2:
        sig = " ".join(sorted(core_toks)[:3])
        keys.add(("s", sig))

    for n in nums:
        if f2 and len(n) >= 2:
            keys.add(("nf", n, f2))

    for t in addr_toks_f:
        keys.add(("a", t))

    return keys


def extract_blocking_keys_parsed(r: IndiaParsed) -> set:
    keys = set()
    f2 = r.name_norm[:2] if len(r.name_norm) >= 2 else r.name_norm

    for p in r.pins:
        keys.add(("p", p))
        if f2:
            keys.add(("pf", p, f2))

    for t in r.core_toks:
        if len(t) >= 4:
            keys.add(("n", t))

    if len(r.core_toks) >= 2:
        sig = " ".join(sorted(r.core_toks)[:3])
        keys.add(("s", sig))

    for n in r.numbers:
        if f2 and len(n) >= 2:
            keys.add(("nf", n, f2))

    for t in r.addr_toks:
        keys.add(("a", t))

    return keys


def query_india_candidates(r: IndiaParsed, index: dict, max_candidates: int = 10) -> list[int]:
    hit_counts = Counter()
    for k in extract_blocking_keys_parsed(r):
        b = index.get(k)
        if b:
            for j in b:
                hit_counts[j] += 1
    if not hit_counts:
        return []
    return [j for j, _ in hit_counts.most_common(max_candidates)]


jw = JaroWinkler.normalized_similarity
lv = Levenshtein.normalized_similarity
tsr = fuzz.token_sort_ratio
tset = fuzz.token_set_ratio
part = fuzz.partial_ratio


def compute_pair_features(a: IndiaParsed, b: IndiaParsed) -> list[float]:
    an, bn = a.name_norm, b.name_norm
    aa, ba = a.addr_norm, b.addr_norm
    has_n = bool(an) and bool(bn)
    has_a = bool(aa) and bool(ba)

    if not has_n:
        n_jw = n_lev = n_sort = n_set = n_part = 0.0
    elif an == bn:
        n_jw = n_lev = n_sort = n_set = n_part = 1.0
    else:
        n_jw = jw(an, bn)
        n_lev = lv(an, bn)
        n_sort = tsr(an, bn) / 100.0
        n_set = tset(an, bn) / 100.0
        n_part = part(an, bn) / 100.0

    if not has_a:
        a_jw = a_lev = a_sort = a_set = a_part = 0.0
    elif aa == ba:
        a_jw = a_lev = a_sort = a_set = a_part = 1.0
    else:
        a_jw = jw(aa, ba)
        a_lev = lv(aa, ba)
        a_sort = tsr(aa, ba) / 100.0
        a_set = tset(aa, ba) / 100.0
        a_part = part(aa, ba) / 100.0

    a_core_set = set(a.core_toks)
    b_core_set = set(b.core_toks)
    core_u = len(a_core_set | b_core_set)
    n_jac = len(a_core_set & b_core_set) / core_u if core_u else 0.0
    n_cont = (len(a_core_set & b_core_set) / min(len(a_core_set), len(b_core_set))) if (a_core_set and b_core_set) else 0.0
    n_len = min(len(an), len(bn)) / max(len(an), len(bn), 1)

    addr_u = len(a.addr_toks | b.addr_toks)
    a_jac = len(a.addr_toks & b.addr_toks) / addr_u if addr_u else 0.0
    a_len = min(len(aa), len(ba)) / max(len(aa), len(ba), 1)

    num_u = len(set(a.numbers) | set(b.numbers))
    num_ov = len(set(a.numbers) & set(b.numbers)) / num_u if num_u else 0.0

    return [
        n_jw, n_lev, n_sort, n_set, n_part, 0.0, n_jac, n_cont, n_len,
        a_jw, a_lev, a_sort, a_set, a_part, 0.0, a_jac, a_len,
        num_ov, len(a.numbers), len(b.numbers),
        1.0 if (a.pins and a.pins & b.pins) else 0.0,
        1.0 if (a.pins and b.pins) else 0.0,
        1.0,
        n_sort * a_sort, min(n_set, a_set), (n_set + a_set) / 2.0,
        n_sort * num_ov if num_ov else 0.0, a_set * n_jw,
        1.0 if b.is_s2 else 0.0,
        1.0 if (a.first_tok and a.first_tok == b.first_tok) else 0.0,
        1.0 if (has_n and an == bn) else 0.0,
        1.0 if (a_core_set and a_core_set == b_core_set) else 0.0
    ]


def main():
    part_dir = Path("dataset/test/partitions")
    m_path = Path("output/matching_results.tsv")
    c_path = Path("output/candidate_pairs.tsv")
    models_dir = Path("models")
    model_file = models_dir / "lgbm_model.pkl"
    thresh_file = models_dir / "thresholds.json"

    log("=" * 70)
    log("LEAN & COOL INDIA INFERENCE — THERMALLY PROTECTED")
    log("=" * 70)

    # 1. Lower process priority on Windows
    if sys.platform == "win32":
        try:
            import ctypes
            # 0x00004000 = BELOW_NORMAL_PRIORITY_CLASS
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
            log("Process priority: BELOW_NORMAL (protects CPU thermal headroom)")
        except Exception as e:
            log(f"Priority notice: {e}")

    # 2. Load trained model & set n_jobs=2
    log(f"Loading LightGBM model from {model_file}...")
    with open(model_file, "rb") as f:
        clf = pickle.load(f)
    clf.set_params(n_jobs=2)
    with open(thresh_file, "r") as f:
        meta = json.load(f)
    best_t = float(meta["threshold"])
    best_mt = float(meta["min_top"])
    log(f"Decision rule: threshold={best_t:.2f}, min_top={best_mt:.2f} (validation F_0.5 = {meta['val_f05']:.4f})")

    # 3. Read done IDs from matching_results.tsv
    done_ids = set()
    with open(m_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t", 1)
            if parts and parts[0]:
                done_ids.add(parts[0])

    log(f"Already completed entities on disk: {len(done_ids):,} (100% safe & intact)")

    # 4. Read ONLY the remaining S1 lines (no re-loading finished entries)
    log("[1/3] Reading remaining India S1 records...")
    remaining_s1_raw = []
    with open(part_dir / "india_s1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] not in done_ids:
                remaining_s1_raw.append((parts[0], parts[1], parts[2]))

    del done_ids; gc.collect()
    n_remaining = len(remaining_s1_raw)
    log(f"Remaining India S1 to score: {n_remaining:,} entities")

    if n_remaining == 0:
        log("All India entities already scored! Proceeding directly to validation.")
    else:
        # 5. Load India Other as lightweight raw tuples (eid, name, addr)
        t_start = time.time()
        log(f"[2/3] Loading India Other records as lightweight raw tuples (takes ~11s, ~1.7 GB RAM)...")
        other_raw = []
        with open(part_dir / "india_other.tsv", "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 3:
                    other_raw.append((parts[0], parts[1], parts[2]))

        log(f"Loaded {len(other_raw):,} Other records in {time.time() - t_start:.1f}s")

        # 6. Build Inverted Index directly from raw records
        t_idx = time.time()
        log("Building Inverted Index (max_bucket=35)...")
        idx_india = defaultdict(list)
        overflow = set()
        max_bucket = 35

        for j, (_, name, addr) in enumerate(other_raw):
            keys = extract_blocking_keys_raw(name, addr)
            for k in keys:
                if k in overflow:
                    continue
                b = idx_india.get(k)
                if b is None:
                    idx_india[k] = [j]
                elif len(b) < max_bucket:
                    b.append(j)
                else:
                    del idx_india[k]
                    overflow.add(k)

        del overflow; gc.collect()
        log(f"Inverted index ready with {len(idx_india):,} keys in {time.time() - t_idx:.1f}s")

        # 7. Stream Inference in gentle 5,000 batches with 3.0s thermal pause
        log(f"[3/3] Streaming inference across {n_remaining:,} entities (batches of 5,000 + 3.0s cooldown)...")
        BATCH_ENTITIES = 5000

        total_matches = 0
        total_singletons = 0
        parsed_other_cache = {}

        def get_parsed_other(j: int) -> IndiaParsed:
            res = parsed_other_cache.get(j)
            if res is None:
                eid, name, addr = other_raw[j]
                res = IndiaParsed(eid, name, addr)
                parsed_other_cache[j] = res
            return res

        with open(m_path, "a", encoding="utf-8") as fm, open(c_path, "a", encoding="utf-8") as fc:
            for b_start in tqdm(range(0, n_remaining, BATCH_ENTITIES), desc="Cool India Scoring"):
                b_end = min(b_start + BATCH_ENTITIES, n_remaining)
                batch_raw = remaining_s1_raw[b_start:b_end]

                # Parse only the current 5,000 S1 records
                batch_s1 = [IndiaParsed(eid, name, addr) for eid, name, addr in batch_raw]

                batch_pairs_feats = []
                entity_cand_info = []

                for r in batch_s1:
                    cands = query_india_candidates(r, idx_india, max_candidates=10)
                    if not cands:
                        entity_cand_info.append((r.id, [], -1, -1))
                    else:
                        cand_ids = [other_raw[j][0] for j in cands]
                        st = len(batch_pairs_feats)
                        for j in cands:
                            b_rec = get_parsed_other(j)
                            batch_pairs_feats.append(compute_pair_features(r, b_rec))
                        en = len(batch_pairs_feats)
                        entity_cand_info.append((r.id, cand_ids, st, en))

                if batch_pairs_feats:
                    X_batch = np.array(batch_pairs_feats, dtype=np.float32)
                    probs = clf.predict_proba(X_batch)[:, 1]
                else:
                    probs = None

                for eid, cand_ids, st, en in entity_cand_info:
                    if not cand_ids or st == -1:
                        fc.write(f"{eid}\t\n")
                        fm.write(f"{eid}\t\n")
                        total_singletons += 1
                    else:
                        fc.write(f"{eid}\t{','.join(cand_ids)}\n")
                        e_probs = probs[st:en]
                        best_p = max(e_probs, default=0.0)
                        if best_p < best_mt:
                            fm.write(f"{eid}\t\n")
                            total_singletons += 1
                        else:
                            accepted = [cid for cid, p in zip(cand_ids, e_probs) if p >= best_t]
                            if not accepted:
                                total_singletons += 1
                            total_matches += len(accepted)
                            fm.write(f"{eid}\t{','.join(sorted(accepted))}\n")

                fm.flush()
                fc.flush()

                # Clean up batch memory & keep cache bounded
                if len(parsed_other_cache) > 200000:
                    parsed_other_cache.clear()

                del batch_s1, batch_pairs_feats, entity_cand_info; gc.collect()

                if (b_start // BATCH_ENTITIES) % 2 == 0 or b_end == n_remaining:
                    total_so_far = 1332516 + b_end
                    pct = (total_so_far / 1732544) * 100
                    log(f"Progress: {total_so_far:,} / 1,732,544 entities ({pct:.1f}%) | Batch {b_end:,}/{n_remaining:,} done")

                # Thermal cooldown pause between batches to keep CPU cool and prevent overheating crashes
                time.sleep(3.0)

        del other_raw, idx_india, remaining_s1_raw, parsed_other_cache; gc.collect()
        log(f"All India entities completed! Matches added: {total_matches:,}, Singletons added: {total_singletons:,}")

    # 8. Official Validation Across the Entire Test Corpus
    log("\n" + "=" * 70)
    log("RUNNING OFFICIAL VALIDATION ACROSS ALL TEST ENTITIES")
    log("=" * 70)

    test_s1_path = Path("dataset/test/test_source1.tsv")
    expected_s1 = set()
    with open(test_s1_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t", 1)
            if parts and parts[0]:
                expected_s1.add(parts[0].strip())

    total_expected = len(expected_s1)
    log(f"Expected test entities from test_source1: {total_expected:,}")

    seen_m = set()
    s1_leak_count = 0
    with open(m_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        if header != ["source1_entity_id", "matched_entity_ids"]:
            log(f"ERROR: Invalid header in matching_results.tsv: {header}")
            return 1
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            seen_m.add(eid)
            if len(parts) > 1 and parts[1]:
                matches = parts[1].split(",")
                for m in matches:
                    if m.startswith("S1-"):
                        s1_leak_count += 1

    log(f"Actual unique entities in matching_results.tsv: {len(seen_m):,}")
    log(f"Self-matches (S1- in matches): {s1_leak_count}")

    if len(seen_m) != total_expected:
        log(f"ERROR: Missing {total_expected - len(seen_m)} entities!")
        return 1

    if s1_leak_count > 0:
        log(f"ERROR: Found {s1_leak_count} self-matches in output!")
        return 1

    log("SUCCESS: 100% test entities covered, exact header match, 0 self-matches!")
    log(f"FILES READY FOR SUBMISSION: {m_path.resolve()} and {c_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
