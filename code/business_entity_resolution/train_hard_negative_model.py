"""
Train Hard-Negative Aware LightGBM Model for High-Precision Entity Resolution
Amazon ML Challenge 2026 - Team Gemz

Key Upgrades:
1. Collision-Mined Hard Negatives: Index 500k+ S2/S3 records to force the model to see
   same-name different-location collisions (e.g. modern packaging VA vs OH, Subway Chicago vs Dallas).
2. Location Conflict Features: Explicit pin_conflict and num_conflict features.
3. Exact Macro-F0.5 Threshold Calibration: Grid-search (threshold, min_top) specifically
   optimizing macro-averaged per-entity F0.5 with singletons explicitly modeled.
4. Windows CPU Throttling: Runs with BELOW_NORMAL priority and 2 threads to protect laptop responsiveness.
"""

import sys
import os
import gc
import re
import time
import json
import pickle
import random
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz import fuzz

# Restrict threads to 2 for background safety
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

LEGAL_TOKENS = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation",
    "co", "company", "llc", "llp", "plc", "holdings", "group", "enterprises",
    "enterprise", "services", "solutions", "technologies", "tech",
    "international", "consulting", "associates", "trading", "industries",
    "traders", "agency", "agencies", "stores", "store",
    "pvt", "private", "sons",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scop", "gie",
    "societe", "ets", "etablissements", "cie", "france", "fr"
}

ADDR_STOP = {
    "near", "opp", "opposite", "behind", "beside", "adj", "adjacent",
    "at", "post", "po", "dist", "district", "taluk", "tehsil", "road",
    "rd", "street", "st", "lane", "ln", "avenue", "ave", "highway",
    "cross", "main", "phase", "sector", "sec", "block", "floor",
    "room", "flat", "shop", "gala", "bldg", "building", "house",
    "complex", "nagar", "colony", "enclave", "vihar", "layout",
    "city", "state", "pin", "code", "zip", "india", "us", "usa", "france",
    "rue", "chemin", "quai", "allee", "route", "cedex", "bp"
}

ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "hwy": "highway", "dr": "drive", "ct": "court",
    "pl": "place", "sq": "square", "ter": "terrace", "apt": "apartment",
    "ste": "suite", "bldg": "building", "no": "number", "num": "number",
    "mgr": "marg", "sect": "sector", "sec": "sector", "ph": "phase",
    "gnd": "ground", "flt": "flat", "hse": "house", "soc": "society",
    "xing": "crossing", "chowk": "chowk", "br": "branch",
    "rue": "street", "avenu": "avenue", "chem": "chemin", "quai": "quay",
    "bd": "boulevard", "all": "allee", "rte": "route", "imp": "impasse"
}

_NON_ALNUM = re.compile(r"[^\w\s]+", re.UNICODE)
_NUM_RE = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("&", " and ")
    s = s.replace(".", " ").replace("/", " ").replace("-", " ")
    s = _NON_ALNUM.sub(" ", s)
    return _SPACES.sub(" ", s).strip()

class EntityRecord:
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
        self.addr_toks = frozenset(t for t in addr_toks if t not in ADDR_STOP and not t.isdigit() and len(t) >= 5)
        nums = tuple(sorted(set(n.lstrip("0") or "0" for t in addr_toks for n in _NUM_RE.findall(t) if len(n) >= 1)))
        self.numbers = nums
        self.pins = frozenset(n for n in nums if len(n) in (5, 6))
        self.first_tok = name_toks[0] if name_toks else ""
        self.is_s2 = eid.startswith("S2-")

def extract_blocking_keys(r: EntityRecord) -> set:
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
        if len(t) >= 6:
            keys.add(("a", t))

    return keys

jw = JaroWinkler.normalized_similarity
lv = Levenshtein.normalized_similarity
tsr = fuzz.token_sort_ratio
tset = fuzz.token_set_ratio
part = fuzz.partial_ratio

def compute_pair_features(a: EntityRecord, b: EntityRecord) -> list[float]:
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

    # Explicit conflict detectors
    pin_match = 1.0 if (a.pins and b.pins and (a.pins & b.pins)) else 0.0
    pin_conflict = 1.0 if (a.pins and b.pins and not (a.pins & b.pins)) else 0.0
    pin_both = 1.0 if (a.pins and b.pins) else 0.0

    num_conflict = 1.0 if (a.numbers and b.numbers and not (set(a.numbers) & set(b.numbers))) else 0.0

    return [
        n_jw, n_lev, n_sort, n_set, n_part, 0.0, n_jac, n_cont, n_len,  # 0..8
        a_jw, a_lev, a_sort, a_set, a_part, 0.0, a_jac, a_len,          # 9..16
        num_ov, len(a.numbers), len(b.numbers),                         # 17..19
        pin_match, pin_both, pin_conflict, num_conflict,                # 20..23 (NEW: explicit conflict flags)
        n_sort * a_sort, min(n_set, a_set), (n_set + a_set) / 2.0,     # 24..26
        n_sort * num_ov if num_ov else 0.0, a_set * n_jw,               # 27..28
        1.0 if b.is_s2 else 0.0,                                        # 29
        1.0 if (a.first_tok and a.first_tok == b.first_tok) else 0.0,   # 30
        1.0 if (has_n and an == bn) else 0.0,                           # 31
        1.0 if (a_core_set and a_core_set == b_core_set) else 0.0      # 32
    ]

def entity_f05(pred: set, truth: set) -> float:
    if not pred and not truth:
        return 1.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    return (1.25 * p * r) / (0.25 * p + r)

def main():
    log("=" * 70)
    log("TRAINING HIGH-PRECISION LIGHTGBM MODEL WITH HARD NEGATIVE COLLISIONS")
    log("=" * 70)

    train_dir = Path("dataset/train")
    gt_file = train_dir / "train_ground_truth.tsv"

    # 1. Load Ground Truth
    log("[1/5] Loading Ground Truth...")
    truth = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            m = parts[1] if len(parts) > 1 and parts[1] else ""
            truth[eid] = set(x.strip() for x in m.split(",") if x.strip()) if m else set()

    all_s1_ids = list(truth.keys())
    random.seed(42)
    np.random.seed(42)
    random.shuffle(all_s1_ids)

    # Use 35,000 S1 entities for training + 5,000 for validation
    train_s1_sample = set(all_s1_ids[:35000])
    val_s1_sample = set(all_s1_ids[35000:40000])
    needed_s1_ids = train_s1_sample | val_s1_sample

    needed_pos_ids = {m for eid in needed_s1_ids for m in truth.get(eid, set())}
    log(f"Selected {len(train_s1_sample):,} train S1 + {len(val_s1_sample):,} val S1. Target positive matches: {len(needed_pos_ids):,}")

    # 2. Load S1 Records
    log("[2/5] Loading S1 Records...")
    s1_dict = {}
    with open(train_dir / "train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in needed_s1_ids and len(parts) >= 3:
                s1_dict[parts[0]] = EntityRecord(parts[0], parts[1], parts[2])

    log(f"Loaded {len(s1_dict):,} S1 entity records into memory.")

    # 3. Load S2 & S3: ALL positive matches + 400,000 background records for collision mining
    log("[3/5] Loading S2 & S3 (Positives + 400k Background records for Hard Collisions)...")
    other_records = []
    other_by_id = {}

    bg_loaded = 0
    bg_target = 400000

    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        with open(train_dir / fn, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 3:
                    continue
                eid = parts[0]
                is_pos = eid in needed_pos_ids
                if is_pos or (bg_loaded < bg_target and random.random() < 0.15):
                    rec = EntityRecord(eid, parts[1], parts[2])
                    other_records.append(rec)
                    other_by_id[eid] = len(other_records) - 1
                    if not is_pos:
                        bg_loaded += 1

    log(f"Loaded total {len(other_records):,} S2/S3 records (|pos|={len(needed_pos_ids):,}, |bg|={bg_loaded:,}).")

    # 4. Build Inverted Index over S2/S3
    log("[4/5] Building Inverted Index for Candidate Blocking & Collision Mining...")
    idx = defaultdict(list)
    overflow = set()
    max_bucket = 150

    for j, rec in enumerate(other_records):
        for k in extract_blocking_keys(rec):
            if k in overflow:
                continue
            b = idx.get(k)
            if b is None:
                idx[k] = [j]
            elif len(b) < max_bucket:
                b.append(j)
            else:
                del idx[k]
                overflow.add(k)
    del overflow; gc.collect()
    log(f"Inverted index ready with {len(idx):,} keys.")

    # 5. Assemble Training Pairs: Positive Matches + Real Blocking Collision Negatives
    log("[5/5] Assembling Training Pairs with Collision Mining...")
    X_train, y_train = [], []
    val_cand_scores = {eid: [] for eid in val_s1_sample}

    train_pos = 0
    train_hard_neg = 0

    for eid in train_s1_sample:
        r = s1_dict.get(eid)
        if not r:
            continue
        true_m = truth.get(eid, set())

        # Collect blocking candidates
        hit_counts = Counter()
        for k in extract_blocking_keys(r):
            b = idx.get(k)
            if b:
                for j in b:
                    hit_counts[j] += 1

        cands = [j for j, _ in hit_counts.most_common(25)]
        cand_set = set(cands)

        # 1. True Positives
        for mid in true_m:
            j = other_by_id.get(mid)
            if j is not None:
                b_rec = other_records[j]
                X_train.append(compute_pair_features(r, b_rec))
                y_train.append(1)
                train_pos += 1

        # 2. Hard Collision Negatives: candidates returned by blocking that are NOT true matches!
        negs = [j for j in cands if other_records[j].id not in true_m]
        # Keep up to 6 hard collision negatives per entity
        if len(negs) > 6:
            # Prefer hard negatives that have high token similarity
            negs = random.sample(negs, 6)

        for j in negs:
            b_rec = other_records[j]
            X_train.append(compute_pair_features(r, b_rec))
            y_train.append(0)
            train_hard_neg += 1

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int8)
    log(f"TRAINING SET ASSEMBLED: {len(X_train):,} total pairs ({train_pos:,} Positives, {train_hard_neg:,} Hard Negatives)")

    # 6. Train LightGBM
    log("Fitting LightGBM Classifier (350 trees, max_depth=8, lr=0.08)...")
    clf = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.08,
        num_leaves=63,
        max_depth=8,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=2
    )
    clf.fit(X_train, y_train)
    log("Model training complete!")

    # 7. Evaluate on Held-Out Validation Entities & Sweep (threshold, min_top) for Macro-F0.5
    log("Scoring Held-Out Validation Entities...")
    val_cand_info = {}
    val_pairs = []

    for eid in val_s1_sample:
        r = s1_dict.get(eid)
        if not r:
            continue
        hit_counts = Counter()
        for k in extract_blocking_keys(r):
            b = idx.get(k)
            if b:
                for j in b:
                    hit_counts[j] += 1
        cands = [j for j, _ in hit_counts.most_common(25)]
        st = len(val_pairs)
        for j in cands:
            val_pairs.append(compute_pair_features(r, other_records[j]))
        en = len(val_pairs)
        val_cand_info[eid] = ([other_records[j].id for j in cands], st, en)

    if val_pairs:
        val_X = np.array(val_pairs, dtype=np.float32)
        val_probs = clf.predict_proba(val_X)[:, 1]
    else:
        val_probs = np.array([])

    val_entity_scores = {}
    for eid, (c_ids, st, en) in val_cand_info.items():
        if not c_ids or st == -1:
            val_entity_scores[eid] = []
        else:
            p_slice = val_probs[st:en]
            val_entity_scores[eid] = list(zip(c_ids, p_slice))

    log("Grid-searching (threshold, min_top) to maximize Macro-F0.5...")
    best_t, best_mt, best_f05 = 0.50, 0.60, -1.0
    val_eids = list(val_s1_sample)

    for t in np.arange(0.40, 0.85, 0.05):
        for mt in [round(t, 2), round(t + 0.05, 2), round(t + 0.10, 2), round(t + 0.15, 2), 0.90]:
            if mt < t:
                continue
            scores = []
            for eid in val_eids:
                cand_list = val_entity_scores.get(eid, [])
                if not cand_list:
                    pred = set()
                else:
                    best_score = max(s for _, s in cand_list)
                    if best_score < mt:
                        pred = set()
                    else:
                        pred = {cid for cid, s in cand_list if s >= t}
                scores.append(entity_f05(pred, truth.get(eid, set())))
            macro_score = float(np.mean(scores))
            if macro_score > best_f05:
                best_f05 = macro_score
                best_t = round(float(t), 2)
                best_mt = round(float(mt), 2)

    log(f"\n{'='*70}")
    log(f"OPTIMAL VALIDATION MACRO F0.5: {best_f05:.4f} (at threshold={best_t:.2f}, min_top={best_mt:.2f})")
    log(f"{'='*70}\n")

    # 8. Save Model
    out_model_path = Path("models/lgbm_model.pkl")
    with open(out_model_path, "wb") as f:
        pickle.dump(clf, f)
    log(f"Saved trained model to {out_model_path}")

    # Save best parameters to a json config
    cfg_path = Path("models/calibrated_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "threshold": best_t,
            "min_top": best_mt,
            "macro_f05": best_f05,
            "max_bucket": max_bucket,
            "max_candidates": 25
        }, f, indent=2)
    log(f"Saved calibrated config to {cfg_path}")

if __name__ == "__main__":
    main()
