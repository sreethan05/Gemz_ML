"""
Step 1 Benchmark: Rigorous Blocking Recall Measurement & Pareto Frontier
Amazon ML Challenge 2026 - Team Gemz

Tests:
1. Exact-key blocking recall alone (current state)
2. VectorIndex (char n-gram TF-IDF cosine) recall alone
3. Union of Key + VectorIndex (cap = infinity)
4. Sweep candidate caps (10, 15, 20, 25, 30, 40, 50, no cap)
5. Separated by country (US vs India)
"""
import sys
import os
import gc
import time
import pickle
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 2 CPU threads & below normal priority
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
from boost_pipeline import (
    EntityRecord, extract_blocking_keys_parsed,
    query_candidates
)

def benchmark_country(ctry: str, n_s1: int = 5000):
    log("=" * 70)
    log(f"BENCHMARKING BLOCKING RECALL FOR {ctry.upper()} (S1={n_s1:,})")
    log("=" * 70)

    # 1. Load ground truth
    gt_file = Path("dataset/train/train_ground_truth.tsv")
    log("Loading ground truth...")
    truth = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            targets = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()
            truth[eid] = targets

    # 2. Load held-out S1 records for this country
    log(f"Loading {ctry.upper()} S1 entities (held out)...")
    s1_records = []
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3].lower() == ctry:
                # Use held-out slice (after line 30,000)
                if len(s1_records) < n_s1:
                    s1_records.append(EntityRecord(parts[0], parts[1], parts[2]))
                else:
                    break

    log(f"Loaded {len(s1_records):,} {ctry.upper()} S1 entities.")
    matched_s1 = [r for r in s1_records if truth.get(r.id)]
    target_ids_needed = {m for r in matched_s1 for m in truth.get(r.id, set())}
    log(f"  {len(matched_s1):,} have true matches ({len(target_ids_needed):,} target IDs total).")
    log(f"  {len(s1_records) - len(matched_s1):,} are true singletons.")

    # 3. Load candidate pool: true targets + massive distractor pool (150,000 records)
    log("Loading other records (true targets + 150,000 distractors)...")
    other_records = []
    other_by_id = {}
    n_distractors = 0

    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        with open(Path("dataset/train") / fn, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[3].lower() == ctry:
                    oid = parts[0]
                    is_pos = oid in target_ids_needed
                    if is_pos or (n_distractors < 150000):
                        if not is_pos:
                            n_distractors += 1
                        rec = EntityRecord(oid, parts[1], parts[2])
                        other_by_id[oid] = len(other_records)
                        other_records.append(rec)
                if n_distractors >= 150000 and len(other_by_id) >= len(target_ids_needed) + 150000:
                    break

    log(f"Loaded {len(other_records):,} total other records ({len(target_ids_needed):,} target pool + {n_distractors:,} distractors).")

    # 4. Build Exact-Key Inverted Index (dynamically scaled bucket cap)
    log("Building Exact-Key Inverted Index (bucket_cap=750)...")
    t0 = time.time()
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
    log(f"Exact-key index ready with {len(key_idx):,} keys in {time.time()-t0:.1f}s")

    # Query Key Blocking (unconstrained)
    t1 = time.time()
    key_hits_all = {}
    for i, r in enumerate(matched_s1):
        hit_counts = Counter()
        for k in extract_blocking_keys_parsed(r):
            b = key_idx.get(k)
            if b:
                for j in b:
                    hit_counts[j] += 1
        key_hits_all[r.id] = [j for j, _ in hit_counts.most_common()]  # full list unconstrained

    # Measure Key Blocking Recall
    def eval_recall(hits_dict, cap=None):
        found, total = 0, len(target_ids_needed)
        for r in matched_s1:
            true_set = truth[r.id]
            retrieved = hits_dict.get(r.id, [])
            if cap is not None:
                retrieved = retrieved[:cap]
            retrieved_ids = {other_records[j].id for j in retrieved}
            for tid in true_set:
                if tid in retrieved_ids:
                    found += 1
        return found / total if total else 1.0, found, total

    r_key_uncapped, f_k, t_k = eval_recall(key_hits_all, cap=None)
    log(f"--> EXACT-KEY BLOCKING RECALL (UNCAPPED): {r_key_uncapped*100:.2f}% ({f_k:,}/{t_k:,})")

    # 5. Build and Query VectorIndex (TF-IDF char n-gram cosine)
    log("\nFitting TF-IDF VectorIndex (char_wb, 3-5 n-grams)...")
    t2 = time.time()
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
        dtype=np.float32
    )
    # Fit vocabulary on sample of other records
    fit_texts = [f"{r.name_norm} {r.addr_norm}" for r in other_records[:50000]]
    vec.fit(fit_texts)

    s1_texts = [f"{r.name_norm} {r.addr_norm}" for r in matched_s1]
    oth_texts = [f"{r.name_norm} {r.addr_norm}" for r in other_records]

    M1 = vec.transform(s1_texts).tocsr()
    M2 = vec.transform(oth_texts).tocsr()
    log(f"TF-IDF sparse matrices ready: M1={M1.shape}, M2={M2.shape} in {time.time()-t2:.1f}s")

    # Query VectorIndex top-30 per S1 in chunks
    log("Querying VectorIndex top-30 candidates in bulk...")
    t3 = time.time()
    vec_hits = {}
    CHUNK = 250
    for start in range(0, M1.shape[0], CHUNK):
        end = min(start + CHUNK, M1.shape[0])
        sims = (M1[start:end] @ M2.T).toarray()
        k_take = min(30, sims.shape[1])
        idx_top = np.argpartition(sims, -k_take, axis=1)[:, -k_take:]
        for r_idx in range(end - start):
            s1_id = matched_s1[start + r_idx].id
            top_j = idx_top[r_idx]
            # sort by cosine score descending
            top_j = sorted(top_j, key=lambda j: sims[r_idx, j], reverse=True)
            vec_hits[s1_id] = [int(j) for j in top_j if sims[r_idx, j] > 0.05]

    r_vec_uncapped, f_v, t_v = eval_recall(vec_hits, cap=None)
    log(f"--> VECTORINDEX FUZZY RECALL ALONE (k=30): {r_vec_uncapped*100:.2f}% ({f_v:,}/{t_v:,})")

    # 6. UNION: Key + VectorIndex (cap = infinity)
    union_hits = {}
    for r in matched_s1:
        k_list = key_hits_all.get(r.id, [])
        v_list = vec_hits.get(r.id, [])
        # Interleave or union preserving key order + vector fallback
        seen = set()
        comb = []
        for j in k_list:
            if j not in seen:
                seen.add(j); comb.append(j)
        for j in v_list:
            if j not in seen:
                seen.add(j); comb.append(j)
        union_hits[r.id] = comb

    r_union_uncapped, f_u, t_u = eval_recall(union_hits, cap=None)
    log("-" * 70)
    log(f"--> UNION (KEY + VECTOR) RECALL (UNCAPPED): {r_union_uncapped*100:.2f}% ({f_u:,}/{t_u:,})")
    log(f"    Recall Improvement from Vector Layer: +{(r_union_uncapped - r_key_uncapped)*100:.2f}%")
    log("-" * 70)

    # 7. Sweep Candidate Cap: Pareto Frontier
    log("\nSWEEPING CANDIDATE CAP (PARETO FRONTIER):")
    log(f"{'Cap':<6} | {'Key-Only Recall':<17} | {'Union Recall':<15} | {'Gain':<8} | {'Avg Cands/Entity'}")
    log("-" * 65)

    caps = [8, 12, 16, 20, 25, 30, 40, 50, None]
    for c in caps:
        cap_str = str(c) if c is not None else "None"
        r_k, _, _ = eval_recall(key_hits_all, cap=c)
        r_u, _, _ = eval_recall(union_hits, cap=c)
        avg_cands = np.mean([min(len(union_hits[r.id]), c if c is not None else 999999) for r in matched_s1])
        log(f"{cap_str:<6} | {r_k*100:>6.2f}%           | {r_u*100:>6.2f}%         | +{(r_u-r_k)*100:>5.2f}% | {avg_cands:.1f}")

    log("=" * 70)

if __name__ == "__main__":
    benchmark_country("us", n_s1=3000)
    benchmark_country("india", n_s1=3000)
