"""
Step 2 A/B Test: Restoring TF-IDF Cosine Features (Slots 5 & 14)
Amazon ML Challenge 2026 - Team Gemz

A/B Test:
- Model A: Slots 5 & 14 hardcoded to 0.0 (old setup)
- Model B: Slots 5 & 14 restored with vectorized char n-gram TF-IDF cosine
Controlled split on real training data (US & India).
"""
import sys
import os
import time
import pickle
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

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

sys.path.append(str(Path(__file__).resolve().parent))
from boost_pipeline import EntityRecord, extract_blocking_keys_parsed

def entity_f05(pred: set, truth: set) -> float:
    if not pred and not truth:
        return 1.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    return (1.25 * p * r) / (0.25 * p + r)

jw = JaroWinkler.normalized_similarity
lv = Levenshtein.normalized_similarity
tsr = fuzz.token_sort_ratio
tset = fuzz.token_set_ratio
part = fuzz.partial_ratio

def compute_base_features(a: EntityRecord, b: EntityRecord, n_cos: float = 0.0, a_cos: float = 0.0) -> list[float]:
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

    return [
        n_jw, n_lev, n_sort, n_set, n_part, n_cos, n_jac, n_cont, n_len,  # 0..8 (slot 5 is n_cos)
        a_jw, a_lev, a_sort, a_set, a_part, a_cos, a_jac, a_len,          # 9..16 (slot 14 is a_cos)
        num_ov, len(a.numbers), len(b.numbers),                         # 17..19
        pin_match, pin_both, pin_conflict, num_conflict,                # 20..23
        n_sort * a_sort, min(n_set, a_set), (n_set + a_set) / 2.0,     # 24..26
        n_sort * num_ov if num_ov else 0.0, a_set * n_jw,               # 27..28
        1.0 if b.is_s2 else 0.0,                                        # 29
        1.0 if (a.first_tok and a.first_tok == b.first_tok) else 0.0,   # 30
        1.0 if (has_n and an == bn) else 0.0,                           # 31
        1.0 if (a_core_set and a_core_set == b_core_set) else 0.0      # 32
    ]

def run_ab_test():
    log("=" * 70)
    log("STEP 2 CONTROLLED A/B TEST: TF-IDF COSINE FEATURES ON VS OFF")
    log("=" * 70)

    # 1. Load ground truth
    truth = {}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            truth[parts[0]] = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()

    # 2. Load 5,000 training S1 and 2,500 validation S1 (held out)
    train_s1, val_s1 = [], []
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 3:
                rec = EntityRecord(parts[0], parts[1], parts[2])
                if len(train_s1) < 5000:
                    train_s1.append(rec)
                elif len(val_s1) < 2500:
                    val_s1.append(rec)
                else:
                    break

    log(f"Data Split: {len(train_s1):,} Train S1 | {len(val_s1):,} Validation S1.")
    all_needed_targets = {m for r in train_s1 + val_s1 for m in truth.get(r.id, set())}

    # 3. Load other records (targets + 100k distractors)
    log("Loading other records (targets + 100k distractors)...")
    other_records = []
    other_by_id = {}
    n_dist = 0
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        with open(Path("dataset/train") / fn, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 3:
                    oid = parts[0]
                    is_pos = oid in all_needed_targets
                    if is_pos or n_dist < 100000:
                        if not is_pos:
                            n_dist += 1
                        rec = EntityRecord(oid, parts[1], parts[2])
                        other_by_id[oid] = len(other_records)
                        other_records.append(rec)
                if n_dist >= 100000 and len(other_by_id) >= len(all_needed_targets) + 100000:
                    break

    log(f"Total Other pool: {len(other_records):,} records.")

    # 4. Fit Vectorizers
    log("Fitting Name and Address TF-IDF Vectorizers (char_wb, 3-5)...")
    t0 = time.time()
    kw = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40000, dtype=np.float32)
    name_vec = TfidfVectorizer(**kw).fit([r.name_norm or " " for r in other_records[:50000]])
    addr_vec = TfidfVectorizer(**kw).fit([r.addr_norm or " " for r in other_records[:50000]])

    log("Transforming records to sparse matrices...")
    train_name_M = name_vec.transform([r.name_norm or " " for r in train_s1]).tocsr()
    train_addr_M = addr_vec.transform([r.addr_norm or " " for r in train_s1]).tocsr()
    val_name_M = name_vec.transform([r.name_norm or " " for r in val_s1]).tocsr()
    val_addr_M = addr_vec.transform([r.addr_norm or " " for r in val_s1]).tocsr()
    oth_name_M = name_vec.transform([r.name_norm or " " for r in other_records]).tocsr()
    oth_addr_M = addr_vec.transform([r.addr_norm or " " for r in other_records]).tocsr()
    log(f"Sparse representations ready in {time.time()-t0:.1f}s.")

    # Build Exact-Key Index for candidates
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
                del key_idx[k]; overflow.add(k)

    def generate_pairs_and_cosines(s1_recs, s1_name_mat, s1_addr_mat, max_c=25):
        pairs = []
        labels = []
        for i, r in enumerate(s1_recs):
            true_set = truth.get(r.id, set())
            hit_counts = Counter()
            for k in extract_blocking_keys_parsed(r):
                b = key_idx.get(k)
                if b:
                    for j in b:
                        hit_counts[j] += 1
            cands = [j for j, _ in hit_counts.most_common(max_c)]
            # ensure true targets in train
            for tid in true_set:
                j = other_by_id.get(tid)
                if j is not None and j not in cands:
                    cands.append(j)
            for j in cands:
                pairs.append((i, j))
                labels.append(1 if other_records[j].id in true_set else 0)

        n_p = len(pairs)
        ai = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=n_p)
        bi = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=n_p)
        # Vectorized sparse dot products
        name_cosines = np.asarray(s1_name_mat[ai].multiply(oth_name_M[bi]).sum(axis=1)).ravel()
        addr_cosines = np.asarray(s1_addr_mat[ai].multiply(oth_addr_M[bi]).sum(axis=1)).ravel()
        return pairs, labels, name_cosines, addr_cosines

    log("Building train pairs & vectorized cosines...")
    tr_pairs, tr_labels, tr_ncos, tr_acos = generate_pairs_and_cosines(train_s1, train_name_M, train_addr_M)
    log(f"Train pairs: {len(tr_pairs):,} ({sum(tr_labels):,} pos, {len(tr_labels)-sum(tr_labels):,} neg).")

    log("Building validation pairs & vectorized cosines...")
    val_pairs, val_labels, val_ncos, val_acos = generate_pairs_and_cosines(val_s1, val_name_M, val_addr_M)
    log(f"Val pairs: {len(val_pairs):,}.")

    # Featurize Model A (0.0 in slots 5 & 14) vs Model B (real cosines)
    log("Computing feature matrices for Model A (TF-IDF Off) and Model B (TF-IDF On)...")
    X_train_A, X_train_B = [], []
    for k, (i, j) in enumerate(tr_pairs):
        f_A = compute_base_features(train_s1[i], other_records[j], n_cos=0.0, a_cos=0.0)
        f_B = compute_base_features(train_s1[i], other_records[j], n_cos=float(tr_ncos[k]), a_cos=float(tr_acos[k]))
        X_train_A.append(f_A)
        X_train_B.append(f_B)

    X_val_A, X_val_B = [], []
    for k, (i, j) in enumerate(val_pairs):
        f_A = compute_base_features(val_s1[i], other_records[j], n_cos=0.0, a_cos=0.0)
        f_B = compute_base_features(val_s1[i], other_records[j], n_cos=float(val_ncos[k]), a_cos=float(val_acos[k]))
        X_val_A.append(f_A)
        X_val_B.append(f_B)

    y_train = np.array(tr_labels, dtype=np.int8)

    # Train Model A
    log("\nFitting Model A (Slots 5 & 14 = 0.0)...")
    clf_A = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=63, random_state=42, n_jobs=2)
    clf_A.fit(np.array(X_train_A, dtype=np.float32), y_train)

    # Train Model B
    log("Fitting Model B (Slots 5 & 14 = Real TF-IDF Cosine)...")
    clf_B = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=63, random_state=42, n_jobs=2)
    clf_B.fit(np.array(X_train_B, dtype=np.float32), y_train)

    # Evaluate on Validation Set
    probs_A = clf_A.predict_proba(np.array(X_val_A, dtype=np.float32))[:, 1]
    probs_B = clf_B.predict_proba(np.array(X_val_B, dtype=np.float32))[:, 1]

    # Group validation pairs by entity
    val_cand_info = defaultdict(list)
    for k, (i, j) in enumerate(val_pairs):
        val_cand_info[i].append((other_records[j].id, probs_A[k], probs_B[k]))

    def evaluate_model(pred_idx, thresh=0.90, min_top=0.90):
        scores = []
        for i, r in enumerate(val_s1):
            true_set = truth.get(r.id, set())
            cand_probs = [(cid, p[pred_idx]) for cid, *p in val_cand_info[i]]
            best_p = max((p for _, p in cand_probs), default=0.0)
            if best_p < min_top:
                pred_set = set()
            else:
                pred_set = {cid for cid, p in cand_probs if p >= thresh}
            scores.append(entity_f05(pred_set, true_set))
        return float(np.mean(scores))

    log("\n" + "=" * 65)
    log("A/B TEST RESULTS ON HELD-OUT VALIDATION SET:")
    log(f"{'Threshold':<10} | {'Model A (TF-IDF Off)':<22} | {'Model B (TF-IDF On)':<22} | {'Delta'}")
    log("-" * 65)
    for t in [0.70, 0.80, 0.85, 0.90, 0.92]:
        sc_A = evaluate_model(pred_idx=0, thresh=t, min_top=t)
        sc_B = evaluate_model(pred_idx=1, thresh=t, min_top=t)
        log(f"{t:<10.2f} | {sc_A:>8.4f}                 | {sc_B:>8.4f}                 | {sc_B - sc_A:>+7.4f}")
    log("=" * 65)

if __name__ == "__main__":
    run_ab_test()
