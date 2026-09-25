"""
Resume India Partition Inference Engine
Resumes directly from existing completed entities and appends to matching_results.tsv
Team Gemz - Amazon ML Challenge 2026
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
import pandas as pd
import lightgbm as lgb
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz import fuzz

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
    "city", "state", "pin", "code", "zip", "india", "mumbai", "delhi",
    "bengaluru", "bangalore", "kolkata", "chennai", "hyderabad", "pune",
    "ahmedabad", "jaipur", "surat", "lucknow", "kanpur", "nagpur", "indore",
    "bhopal", "patna", "vadodara", "ghaziabad", "ludhiana", "agra", "nashik",
    "faridabad", "meerut", "rajkot", "varanasi", "srinagar", "aurangabad",
    "dhanbad", "amritsar", "allahabad", "ranchi", "howrah", "coimbatore",
    "jabalpur", "gwalior", "vijayawada", "jodhpur", "madurai", "raipur",
    "kota", "guwahati", "chandigarh", "solapur", "hubli", "dharwad",
    "bareilly", "moradabad", "mysore", "gurgaon", "aligarh", "jalandhar"
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


class IndiaRecord:
    __slots__ = (
        'id', 'name_norm', 'addr_norm', 'core_toks', 'addr_toks',
        'numbers', 'pins', 'f2', 'first_tok', 'is_s2'
    )

    def __init__(self, eid: str, name: str, address: str):
        name_norm = clean_text(name)
        addr_raw = clean_text(address)
        addr_toks = [ADDR_ABBREV.get(t, t) for t in addr_raw.split()]
        addr_norm = " ".join(addr_toks)
        name_toks = name_norm.split()

        self.id = eid
        self.name_norm = name_norm
        self.addr_norm = addr_norm
        self.core_toks = tuple(t for t in name_toks if t not in LEGAL_TOKENS and len(t) >= 3)
        self.addr_toks = frozenset(t for t in addr_toks if t not in ADDR_STOP and not t.isdigit() and len(t) >= 6)
        nums = tuple(sorted(set(n.lstrip("0") or "0" for t in addr_toks for n in _NUM_RE.findall(t) if len(n) >= 1)))
        self.numbers = nums
        self.pins = frozenset(n for n in nums if len(n) == 6)
        self.f2 = name_norm[:2] if len(name_norm) >= 2 else name_norm
        self.first_tok = name_toks[0] if name_toks else ""
        self.is_s2 = eid.startswith("S2-")


def extract_india_blocking_keys(r: IndiaRecord) -> set:
    keys = set()
    f2 = r.f2
    ctoks = r.core_toks

    for p in r.pins:
        keys.add(("p", p))
        if f2:
            keys.add(("pf", p, f2))

    for t in ctoks:
        if len(t) >= 4:
            keys.add(("n", t))

    if len(ctoks) >= 2:
        sig = " ".join(sorted(ctoks)[:3])
        keys.add(("s", sig))

    for n in r.numbers:
        if f2 and len(n) >= 2:
            keys.add(("nf", n, f2))

    for t in r.addr_toks:
        keys.add(("a", t))

    return keys


def build_india_index(other_recs: list[IndiaRecord], max_bucket: int = 40):
    index = defaultdict(list)
    overflow = set()

    for j, r in enumerate(other_recs):
        for k in extract_india_blocking_keys(r):
            if k in overflow:
                continue
            b = index.get(k)
            if b is None:
                index[k] = [j]
            elif len(b) < max_bucket:
                b.append(j)
            else:
                del index[k]
                overflow.add(k)

    return index


def query_india_candidates(r: IndiaRecord, index: dict, max_candidates: int = 12) -> list[int]:
    hit_counts = Counter()
    for k in extract_india_blocking_keys(r):
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


def compute_pair_features(a: IndiaRecord, b: IndiaRecord) -> list[float]:
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


def load_india_compact(path: Path, skip_ids: set = None) -> list[IndiaRecord]:
    recs = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=400000, quoting=csv.QUOTE_NONE):
        for r in chunk.itertuples(index=False):
            eid = str(r.entity_id).strip()
            if skip_ids is not None and eid in skip_ids:
                continue
            recs.append(IndiaRecord(eid, r.business_name, r.business_address))
    return recs


def main():
    part_dir = Path("dataset/test/partitions")
    m_path = Path("output/matching_results.tsv")
    c_path = Path("output/candidate_pairs.tsv")
    models_dir = Path("models")
    model_file = models_dir / "lgbm_model.pkl"
    thresh_file = models_dir / "thresholds.json"

    log("=" * 70)
    log("RESUMING INDIA PARTITION INFERENCE — MAXIMUM PERFORMANCE")
    log("=" * 70)

    # 1. Load trained model & thresholds
    log(f"Loading model from {model_file}...")
    with open(model_file, "rb") as f:
        clf = pickle.load(f)
    with open(thresh_file, "r") as f:
        meta = json.load(f)
    best_t = float(meta["threshold"])
    best_mt = float(meta["min_top"])
    log(f"Loaded decision rule: threshold={best_t:.2f}, min_top={best_mt:.2f} (validation F_0.5 = {meta['val_f05']:.4f})")

    # 2. Inspect existing output files
    existing_eids = set()
    with open(m_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts and parts[0]:
                existing_eids.add(parts[0])

    log(f"Existing completed entities in matching_results.tsv: {len(existing_eids):,}")

    t_start = time.time()

    # 3. Load remaining India S1 entities (skipping already processed)
    log("[1/3] Loading India S1 records (filtering out already completed)...")
    s1_india = load_india_compact(part_dir / "india_s1.tsv", skip_ids=existing_eids)
    log(f"Remaining India S1 to score: {len(s1_india):,} entities (already completed: {len(existing_eids):,})")

    if not s1_india:
        log("All India entities are already completed!")
    else:
        # Load Other records
        log("Loading India Other records...")
        other_india = load_india_compact(part_dir / "india_other.tsv")
        log(f"Loaded {len(other_india):,} Other records in {(time.time() - t_start):.1f}s")

        # 4. Build Sharp Bounded Index (max_bucket=40 for extreme speed)
        t_idx = time.time()
        log("[2/3] Building Sharp Bounded Inverted Index (max_bucket=40)...")
        idx_india = build_india_index(other_india, max_bucket=40)
        log(f"Inverted index built with {len(idx_india):,} keys in {(time.time() - t_idx):.1f}s")

        # 5. Stream Inference
        log(f"[3/3] Streaming inference across remaining {len(s1_india):,} entities...")
        BATCH_ENTITIES = 25000
        n_s1 = len(s1_india)

        total_matches = 0
        total_singletons = 0

        with open(m_path, "a", encoding="utf-8") as fm, open(c_path, "a", encoding="utf-8") as fc:
            for b_start in tqdm(range(0, n_s1, BATCH_ENTITIES), desc="Resuming INDIA"):
                b_end = min(b_start + BATCH_ENTITIES, n_s1)
                batch_s1 = s1_india[b_start:b_end]

                batch_pairs_feats = []
                entity_cand_info = []

                for r in batch_s1:
                    cands = query_india_candidates(r, idx_india, max_candidates=12)
                    if not cands:
                        fm.write(f"{r.id}\t\n")
                        fc.write(f"{r.id}\t\n")
                        total_singletons += 1
                    else:
                        cand_ids = [other_india[j].id for j in cands]
                        st = len(batch_pairs_feats)
                        for j in cands:
                            batch_pairs_feats.append(compute_pair_features(r, other_india[j]))
                        en = len(batch_pairs_feats)
                        entity_cand_info.append((r.id, cand_ids, st, en))

                if batch_pairs_feats:
                    X_batch = np.array(batch_pairs_feats, dtype=np.float32)
                    probs = clf.predict_proba(X_batch)[:, 1]

                    for eid, cand_ids, st, en in entity_cand_info:
                        e_probs = probs[st:en]
                        best_p = max(e_probs, default=0.0)
                        fc.write(f"{eid}\t{','.join(cand_ids)}\n")
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

        del s1_india, other_india, idx_india; gc.collect()
        log(f"All India entities completed! Matches added: {total_matches:,}, Singletons added: {total_singletons:,}")

    # 6. Official Validation
    log("\n" + "=" * 70)
    log("RUNNING OFFICIAL VALIDATION ACROSS ALL 1,714,000 TEST ENTITIES")
    log("=" * 70)

    test_s1_path = Path("dataset/test/test_source1.tsv")
    expected_s1 = set()
    for chunk in pd.read_csv(test_s1_path, sep="\t", dtype=str, usecols=["entity_id"], chunksize=300000):
        for eid in chunk["entity_id"]:
            expected_s1.add(str(eid).strip())

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
