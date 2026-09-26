"""
Test Cosine Trimming vs Hit-Count Ranking on India Blocking Recall
Amazon ML Challenge 2026 - Team Gemz
"""
import sys
import os
import time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

sys.path.append(str(Path(__file__).resolve().parent))
from boost_pipeline import EntityRecord, extract_blocking_keys_parsed

def run_trim_test(n_s1=2000):
    log("Loading truth and India S1...")
    truth = {}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            truth[parts[0]] = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()

    s1_records = []
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3].lower() == "india":
                s1_records.append(EntityRecord(parts[0], parts[1], parts[2]))
                if len(s1_records) >= n_s1:
                    break

    matched_s1 = [r for r in s1_records if truth.get(r.id)]
    target_ids_needed = {m for r in matched_s1 for m in truth[r.id]}
    log(f"India S1: {len(matched_s1)} matched entities, {len(target_ids_needed)} target IDs.")

    # Load targets + 80k distractors
    other_records = []
    other_by_id = {}
    n_dist = 0
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        with open(Path("dataset/train") / fn, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[3].lower() == "india":
                    oid = parts[0]
                    is_pos = oid in target_ids_needed
                    if is_pos or n_dist < 80000:
                        if not is_pos:
                            n_dist += 1
                        rec = EntityRecord(oid, parts[1], parts[2])
                        other_by_id[oid] = len(other_records)
                        other_records.append(rec)
                if n_dist >= 80000 and len(other_by_id) >= len(target_ids_needed) + 80000:
                    break

    log(f"Other records: {len(other_records)} loaded.")

    # Build exact key index
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

    # Build VectorIndex
    log("Fitting TF-IDF VectorIndex...")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40000, dtype=np.float32)
    vec.fit([f"{r.name_norm} {r.addr_norm}" for r in other_records[:40000]])
    M1 = vec.transform([f"{r.name_norm} {r.addr_norm}" for r in matched_s1]).tocsr()
    M2 = vec.transform([f"{r.name_norm} {r.addr_norm}" for r in other_records]).tocsr()

    log("Computing union and comparing ranking strategies...")
    # Gather union candidates for each S1
    union_cands_dict = {}
    key_ranked_dict = {}

    for i, r in enumerate(matched_s1):
        hit_counts = Counter()
        for k in extract_blocking_keys_parsed(r):
            b = key_idx.get(k)
            if b:
                for j in b:
                    hit_counts[j] += 1
        k_list = [j for j, _ in hit_counts.most_common()]
        key_ranked_dict[r.id] = k_list
        union_cands_dict[r.id] = set(k_list)

    # Add top-25 vector candidates to union
    CHUNK = 250
    for start in range(0, M1.shape[0], CHUNK):
        end = min(start + CHUNK, M1.shape[0])
        sims = (M1[start:end] @ M2.T).toarray()
        k_take = min(25, sims.shape[1])
        idx_top = np.argpartition(sims, -k_take, axis=1)[:, -k_take:]
        for r_idx in range(end - start):
            s1_id = matched_s1[start + r_idx].id
            top_j = idx_top[r_idx]
            for j in top_j:
                if sims[r_idx, j] > 0.05:
                    union_cands_dict[s1_id].add(int(j))

    # Now compare:
    # Strategy A: Crude integer hit count ranking
    # Strategy B: TF-IDF Cosine Score ranking (VectorIndex.trim style)
    cosine_ranked_dict = {}
    for i, r in enumerate(matched_s1):
        cands = list(union_cands_dict[r.id])
        if not cands:
            cosine_ranked_dict[r.id] = []
            continue
        # Compute exact cosine scores for the candidates
        prod = (M1[i] @ M2[cands].T).toarray().ravel()
        # Sort cands by descending cosine score
        sorted_indices = np.argsort(-prod)
        cosine_ranked_dict[r.id] = [cands[idx] for idx in sorted_indices]

    def measure(ranked_dict, cap):
        found, total = 0, len(target_ids_needed)
        for r in matched_s1:
            retrieved_ids = {other_records[j].id for j in ranked_dict[r.id][:cap]}
            for tid in truth[r.id]:
                if tid in retrieved_ids:
                    found += 1
        return found / total * 100

    log("\n" + "=" * 65)
    log(f"{'Cap':<6} | {'Crude Hit-Count Recall':<24} | {'Cosine-Ranked Recall':<22} | {'Gain'}")
    log("-" * 65)
    for cap in [8, 12, 16, 20, 25, 30, 40, None]:
        cap_str = str(cap) if cap is not None else "Uncapped"
        rec_hit = measure(key_ranked_dict, cap)
        rec_cos = measure(cosine_ranked_dict, cap)
        log(f"{cap_str:<6} | {rec_hit:>6.2f}%                  | {rec_cos:>6.2f}%                | +{rec_cos - rec_hit:>5.2f}%")
    log("=" * 65)

if __name__ == "__main__":
    run_trim_test(2000)
