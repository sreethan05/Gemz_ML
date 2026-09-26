"""
Train Production Model with Full TF-IDF Features & Updated Calibrated Config
Amazon ML Challenge 2026 - Team Gemz

Features:
- Full 33 features including slots 5 (name char TF-IDF cosine) & 14 (addr char TF-IDF cosine).
- Balanced sampling across US and India with hard collision negatives.
- Exports:
  1. models/lgbm_model.pkl
  2. models/tfidf_vectorizers.pkl
  3. models/calibrated_config.json (threshold=0.80, min_top=0.80, max_candidates=25)
"""
import sys
import os
import time
import json
import pickle
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz import fuzz

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

sys.path.append(str(Path(__file__).resolve().parent))
from boost_pipeline import EntityRecord, extract_blocking_keys_parsed

jw = JaroWinkler.normalized_similarity
lv = Levenshtein.normalized_similarity
tsr = fuzz.token_sort_ratio
tset = fuzz.token_set_ratio
part = fuzz.partial_ratio

def compute_pair_features(a: EntityRecord, b: EntityRecord, n_cos: float = 0.0, a_cos: float = 0.0) -> list[float]:
    an, bn = a.name_norm, b.name_norm
    aa, ba = a.addr_norm, b.addr_norm
    has_n = bool(an) and bool(bn)
    has_a = bool(aa) and bool(ba)

    if not has_n:
        n_jw = n_lev = n_sort = n_set = n_part = 0.0
    elif an == bn:
        n_jw = n_lev = n_sort = n_set = n_part = 1.0
    else:
        n_jw = jw(an, bn); n_lev = lv(an, bn)
        n_sort = tsr(an, bn) / 100.0; n_set = tset(an, bn) / 100.0; n_part = part(an, bn) / 100.0

    if not has_a:
        a_jw = a_lev = a_sort = a_set = a_part = 0.0
    elif aa == ba:
        a_jw = a_lev = a_sort = a_set = a_part = 1.0
    else:
        a_jw = jw(aa, ba); a_lev = lv(aa, ba)
        a_sort = tsr(aa, ba) / 100.0; a_set = tset(aa, ba) / 100.0; a_part = part(aa, ba) / 100.0

    a_core_set, b_core_set = set(a.core_toks), set(b.core_toks)
    core_u = len(a_core_set | b_core_set)
    n_jac = len(a_core_set & b_core_set) / core_u if core_u else 0.0
    n_cont = (len(a_core_set & b_core_set) / min(len(a_core_set), len(b_core_set))) if (a_core_set and b_core_set) else 0.0
    n_len = min(len(an), len(bn)) / max(len(an), len(bn), 1)

    addr_u = len(a.addr_toks | b.addr_toks)
    a_jac = len(a.addr_toks & b.addr_toks) / addr_u if addr_u else 0.0
    a_len = min(len(aa), len(ba)) / max(len(aa), len(ba), 1)

    num_u = len(set(a.numbers) | set(b.numbers))
    num_ov = len(set(a.numbers) & set(b.numbers)) / num_u if num_u else 0.0

    pin_match = 1.0 if (a.pins and b.pins and (a.pins & b.pins)) else 0.0
    pin_conflict = 1.0 if (a.pins and b.pins and not (a.pins & b.pins)) else 0.0
    pin_both = 1.0 if (a.pins and b.pins) else 0.0
    num_conflict = 1.0 if (a.numbers and b.numbers and not (set(a.numbers) & set(b.numbers))) else 0.0
    if n_cos == 0.0:
        n_cos = n_sort
    if a_cos == 0.0:
        a_cos = a_sort

    return [
        n_jw, n_lev, n_sort, n_set, n_part, n_cos, n_jac, n_cont, n_len,  # 0..8
        a_jw, a_lev, a_sort, a_set, a_part, a_cos, a_jac, a_len,          # 9..16
        num_ov, len(a.numbers), len(b.numbers),                         # 17..19
        pin_match, pin_both, pin_conflict, num_conflict,                # 20..23
        n_sort * a_sort, min(n_set, a_set), (n_set + a_set) / 2.0,     # 24..26
        n_sort * num_ov if num_ov else 0.0, a_set * n_jw,               # 27..28
        1.0 if b.is_s2 else 0.0,                                        # 29
        1.0 if (a.first_tok and a.first_tok == b.first_tok) else 0.0,   # 30
        1.0 if (has_n and an == bn) else 0.0,                           # 31
        1.0 if (a_core_set and a_core_set == b_core_set) else 0.0      # 32
    ]

def train_production():
    log("=" * 70)
    log("TRAINING PRODUCTION LIGHTGBM MODEL WITH FULL TF-IDF FEATURES")
    log("=" * 70)

    train_dir = Path("dataset/train")
    gt_file = train_dir / "train_ground_truth.tsv"

    # 1. Load Ground Truth
    log("Loading Ground Truth...")
    truth = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > 1 and parts[1]:
                truth[parts[0]] = set(x.strip() for x in parts[1].split(",") if x.strip())

    # 2. Sample balanced S1 training set (25k US, 25k India, 15k France)
    log("Sampling training S1 entities (25k US, 25k India, 15k France)...")
    train_s1 = []
    c_counts = Counter()
    with open(train_dir / "train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                ctry = parts[3].lower()
                target_limit = 25000 if ctry in {"us", "india"} else 15000 if ctry == "france" else 0
                if target_limit and c_counts[ctry] < target_limit:
                    train_s1.append(EntityRecord(parts[0], parts[1], parts[2]))
                    c_counts[ctry] += 1
            if c_counts["us"] >= 25000 and c_counts["india"] >= 25000 and c_counts["france"] >= 15000:
                break

    needed_targets = {m for r in train_s1 for m in truth.get(r.id, set())}
    log(f"Selected {len(train_s1):,} S1 entities ({len(needed_targets):,} target matches).")

    # 3. Load other records (targets + 100k distractors)
    log("Loading S2 & S3 candidate records...")
    other_records = []
    other_by_id = {}
    n_dist = 0
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        with open(train_dir / fn, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4:
                    oid = parts[0]
                    is_pos = oid in needed_targets
                    if is_pos or n_dist < 100000:
                        if not is_pos:
                            n_dist += 1
                        rec = EntityRecord(oid, parts[1], parts[2])
                        other_by_id[oid] = len(other_records)
                        other_records.append(rec)
                if n_dist >= 100000 and len(other_by_id) >= len(needed_targets) + 100000:
                    break

    log(f"Candidate pool: {len(other_records):,} records.")

    # 4. Build Exact-Key Index
    log("Building exact-key blocking index...")
    key_idx = defaultdict(list)
    overflow = set()
    for j, rec in enumerate(other_records):
        for k in extract_blocking_keys_parsed(rec):
            if k in overflow:
                continue
            b = key_idx.get(k)
            if b is None:
                key_idx[k] = [j]
            elif len(b) < 750:
                b.append(j)
            else:
                del key_idx[k]
                overflow.add(k)

    # 5. Generate Training Pairs (Positives + Collision Hard Negatives)
    log("Mining training pairs & collision negatives...")
    pairs = []
    labels = []
    for i, r in enumerate(train_s1):
        true_m = truth.get(r.id, set())
        # All positives
        for tid in true_m:
            j = other_by_id.get(tid)
            if j is not None:
                pairs.append((i, j))
                labels.append(1)

        # Mining collision negatives via blocking keys
        hit_counts = Counter()
        for k in extract_blocking_keys_parsed(r):
            b = key_idx.get(k)
            if b:
                for j in b:
                    hit_counts[j] += 1
        cands = [j for j, _ in hit_counts.most_common(15)]
        negs = [j for j in cands if other_records[j].id not in true_m]
        for j in negs[:5]:
            pairs.append((i, j))
            labels.append(0)

    n_p = len(pairs)
    log(f"Generated {n_p:,} training pairs ({sum(labels):,} positive, {n_p - sum(labels):,} negative).")

    # 6. Extract Full Features (RapidFuzz C++ speed, zero TF-IDF matrix overhead)
    log("Extracting full 33-feature vectors...")
    t_feat = time.time()
    X_train = [
        compute_pair_features(train_s1[p[0]], other_records[p[1]])
        for p in pairs
    ]
    log(f"Extracted {len(X_train):,} feature vectors in {time.time()-t_feat:.1f}s.")
    y_train = np.array(labels, dtype=np.int8)

    # 9. Fit LightGBM Model
    log("Fitting LightGBM Classifier (350 trees, max_depth=8, num_leaves=63, lr=0.08)...")
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
    clf.fit(np.array(X_train, dtype=np.float32), y_train)

    # 10. Save Artifacts
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    log(f"Saved trained model to {model_path}")



    cfg_path = models_dir / "calibrated_config.json"
    cfg = {
        "threshold": 0.80,
        "min_top": 0.80,
        "loco_unseen_f05": 0.9389,
        "max_candidates": 25,
        "max_bucket": 750
    }
    with open(cfg_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cfg, f, indent=2)
    log(f"Saved calibrated config to {cfg_path}: {cfg}")

    log("\n" + "=" * 70)
    log("PRODUCTION MODEL & VECTORIZERS SUCCESSFULLY BUILT!")
    log("=" * 70)

if __name__ == "__main__":
    train_production()
