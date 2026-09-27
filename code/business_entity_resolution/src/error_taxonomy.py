"""
Step 5: Error Taxonomy & False Positive Pattern Analysis
Amazon ML Challenge 2026 - Team Gemz

Buckets holdout outcomes into 4 exact categories:
1. Recall failure: True match never blocked
2. Model failure: True match blocked but scored below threshold
3. Precision failure: False Positive (wrong entity accepted)
4. Correct match / Correct singleton

Inspects Category 3 patterns:
- Shared exact address, completely different business name (mall/shopping center/office complex)
- Identical business name, different city/state (franchise/chains)
- Typo/near-duplicate vs genuinely different business
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

jw = JaroWinkler.normalized_similarity
lv = Levenshtein.normalized_similarity
tsr = fuzz.token_sort_ratio
tset = fuzz.token_set_ratio
part = fuzz.partial_ratio

def compute_features(a: EntityRecord, b: EntityRecord, n_cos: float = 0.0, a_cos: float = 0.0) -> list[float]:
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

def run_taxonomy(ctry="india", n_val=2000, thresh=0.90, min_top=0.90):
    log("=" * 70)
    log(f"ERROR TAXONOMY & FAILURE ANALYSIS FOR {ctry.upper()} (N={n_val:,})")
    log("=" * 70)

    # 1. Load ground truth
    truth = {}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            truth[parts[0]] = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()

    # 2. Load held-out S1
    val_s1 = []
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3].lower() == ctry:
                if i > 25000 and len(val_s1) < n_val:
                    val_s1.append(EntityRecord(parts[0], parts[1], parts[2]))
                elif len(val_s1) >= n_val:
                    break

    needed_targets = {m for r in val_s1 for m in truth.get(r.id, set())}
    log(f"Loaded {len(val_s1):,} {ctry.upper()} entities ({len(needed_targets):,} target matches).")

    # 3. Load other records (targets + 100k distractors)
    other_records = []
    other_by_id = {}
    n_dist = 0
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        with open(Path("dataset/train") / fn, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[3].lower() == ctry:
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

    log(f"Candidate pool size: {len(other_records):,} records.")

    # 4. Inverted key index + VectorIndex
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

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40000, dtype=np.float32)
    vec.fit([f"{r.name_norm} {r.addr_norm}" for r in other_records[:40000]])
    M1 = vec.transform([f"{r.name_norm} {r.addr_norm}" for r in val_s1]).tocsr()
    M2 = vec.transform([f"{r.name_norm} {r.addr_norm}" for r in other_records]).tocsr()

    # Load Model B
    with open("models/lgbm_model.pkl", "rb") as f:
        clf = pickle.load(f)
    clf.set_params(n_jobs=2)

    cat1_recall_fail = []    # True match never blocked
    cat2_model_fail = []     # True match blocked but scored < threshold
    cat3_false_pos = []      # Wrong entity accepted
    cat4_correct = 0         # Correct match or correct singleton

    log("Running candidate generation and scoring for taxonomy...")
    for i, r in enumerate(val_s1):
        true_set = truth.get(r.id, set())
        
        # 1. Blocking candidates (Key + Vector union)
        hit_counts = Counter()
        for k in extract_blocking_keys_parsed(r):
            b = key_idx.get(k)
            if b:
                for j in b:
                    hit_counts[j] += 1
        cand_indices = set(j for j, _ in hit_counts.most_common(25))

        # Cosine top-10
        sims = (M1[i] @ M2.T).toarray().ravel()
        top_vec = np.argpartition(sims, -10)[-10:]
        cand_indices.update(int(j) for j in top_vec if sims[j] > 0.05)

        # Candidate IDs
        cand_list = list(cand_indices)
        blocked_tids = {other_records[j].id for j in cand_list}

        # Check Category 1: Recall failure
        missed_by_blocking = true_set - blocked_tids
        if missed_by_blocking:
            cat1_recall_fail.append((r, missed_by_blocking))

        if not cand_list:
            if not true_set:
                cat4_correct += 1
            continue

        # Featurize
        ai = np.full(len(cand_list), i, dtype=np.int64)
        bi = np.array(cand_list, dtype=np.int64)
        n_cos = np.asarray(M1[ai].multiply(M2[bi]).sum(axis=1)).ravel()
        a_cos = np.asarray(M1[ai].multiply(M2[bi]).sum(axis=1)).ravel()

        feats = [compute_features(r, other_records[j], float(n_cos[p_i]), float(a_cos[p_i])) for p_i, j in enumerate(cand_list)]
        probs = clf.predict_proba(np.array(feats, dtype=np.float32))[:, 1]

        # Location veto & address rescue
        for idx_p, k_feat in enumerate(range(len(cand_list))):
            pin_conflict = feats[k_feat][22]
            a_sort = feats[k_feat][11]
            if pin_conflict > 0.5 and a_sort < 0.90:
                probs[idx_p] = 0.0
            elif a_sort >= 0.90 and (feats[k_feat][20] > 0.5 or feats[k_feat][17] >= 0.5):
                probs[idx_p] = max(probs[idx_p], thresh)

        best_p = max(probs, default=0.0)
        if best_p < min_top:
            accepted = set()
        else:
            accepted = {other_records[j].id for j, p in zip(cand_list, probs) if p >= thresh}

        # Check Category 2: Model failure (blocked, but scored < threshold)
        blocked_but_rejected = (true_set & blocked_tids) - accepted
        if blocked_but_rejected:
            cat2_model_fail.append((r, blocked_but_rejected))

        # Check Category 3: False positives (wrong entity accepted)
        false_merges = accepted - true_set
        if false_merges:
            cat3_false_pos.append((r, false_merges, [(other_records[other_by_id[m_id]], p) for m_id, p in zip(blocked_tids, probs) if m_id in false_merges]))

        # Category 4: Correct match / singleton
        if accepted == true_set:
            cat4_correct += 1

    log("\n" + "=" * 70)
    log(f"ERROR TAXONOMY DISTRIBUTION across {len(val_s1):,} {ctry.upper()} entities:")
    log(f"  Category 1 (Recall Failure - True match not blocked):      {len(cat1_recall_fail):>5,} ({len(cat1_recall_fail)/len(val_s1)*100:>5.2f}%)")
    log(f"  Category 2 (Model Failure - Blocked but scored < thresh):   {len(cat2_model_fail):>5,} ({len(cat2_model_fail)/len(val_s1)*100:>5.2f}%)")
    log(f"  Category 3 (Precision Failure - False Positive merges):    {len(cat3_false_pos):>5,} ({len(cat3_false_pos)/len(val_s1)*100:>5.2f}%)")
    log(f"  Category 4 (Perfect Match or Correct Singleton):           {cat4_correct:>5,} ({cat4_correct/len(val_s1)*100:>5.2f}%)")
    log("=" * 70)

    # Print Category 3 (Precision Failures) Case Analysis
    log("\nDEEP DIVE: CATEGORY 3 FALSE POSITIVE EXAMPLES:")
    for idx_ex, (r, false_ids, details) in enumerate(cat3_false_pos[:8], 1):
        log(f"\n[Example {idx_ex}] S1: '{r.name_norm}' | Addr: '{r.addr_norm}'")
        for bad_rec, p in details:
            s_name = jw(r.name_norm, bad_rec.name_norm)
            s_addr = tsr(r.addr_norm, bad_rec.addr_norm) / 100.0
            log(f"   -> F-POS: '{bad_rec.name_norm}' | Addr: '{bad_rec.addr_norm}'")
            log(f"      Score={p:.3f} | Name-JW={s_name:.2f} | Addr-Sort={s_addr:.2f}")

if __name__ == "__main__":
    run_taxonomy("india", n_val=2000)
