"""
Offline Pre-Submission Evaluator
Amazon ML Challenge 2026 - Team Gemz

Evaluates the exact Macro-F0.5 score locally on real held-out ground truth data
BEFORE submitting to the leaderboard portal.
"""
import sys
import os
import gc
import csv
import json
import time
import pickle
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import lightgbm as lgb
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

def entity_f05(pred: set, truth: set) -> float:
    if not pred and not truth:
        return 1.0  # correctly predicted singleton
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Load pipeline dependencies
sys.path.append(str(Path(__file__).resolve().parent))
from boost_pipeline import (
    EntityRecord, extract_blocking_keys_parsed,
    query_candidates, compute_pair_features
)
from enforce_target_1to1 import disambiguate_matching_results

def evaluate_heldout(n_eval: int = 5000):
    log("=" * 70)
    log(f"LOCAL OFFLINE PRE-SUBMISSION EVALUATION ON {n_eval:,} REAL ENTITIES")
    log("=" * 70)

    # 1. Load ground truth
    gt_file = Path("dataset/train/train_ground_truth.tsv")
    log(f"Reading real ground truth from {gt_file}...")
    truth = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0]
            targets = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()
            truth[eid] = targets

    # 2. Pick a held-out set of entities from US & India
    s1_files = [
        ("us", Path("dataset/train/train_source1.tsv")),
    ]
    # Check if partitions exist or read train_source1
    train_s1_path = Path("dataset/train/train_source1.tsv")
    train_oth_paths = [Path("dataset/train/train_source2.tsv"), Path("dataset/train/train_source3.tsv")]

    log("Loading held-out S1 entities (skipping first 50k used in training)...")
    eval_s1 = []
    eval_country = {}
    with open(train_s1_path, "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i < 60000:
                continue  # skip training slice
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 3:
                eval_s1.append(EntityRecord(parts[0], parts[1], parts[2]))
                eval_country[parts[0]] = parts[3].strip().lower() if len(parts) > 3 else ""
            if len(eval_s1) >= n_eval:
                break

    log(f"Loaded {len(eval_s1):,} completely unseen held-out S1 test entities.")
    target_ids_needed = {m for r in eval_s1 for m in truth.get(r.id, set())}
    log(f"Required target match IDs to evaluate recall: {len(target_ids_needed):,}")

    # 3. Load other records (ensuring true matches + 100,000 realistic distractors)
    log("Loading candidate records and building index...")
    other_records = []
    other_by_id = {}
    idx_by_country = defaultdict(lambda: defaultdict(list))
    overflow = set()
    n_distractors = Counter()

    for p in train_oth_paths:
        with open(p, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 3:
                    oid, name, addr = parts[0], parts[1], parts[2]
                    ctry = parts[3].strip().lower() if len(parts) > 3 else ""
                    is_pos = oid in target_ids_needed
                    if is_pos or (ctry in {"us", "india"} and n_distractors[ctry] < 200000):
                        if not is_pos:
                            n_distractors[ctry] += 1
                        rec = EntityRecord(oid, name, addr)
                        j = len(other_records)
                        other_records.append(rec)
                        other_by_id[oid] = j
                        for k in extract_blocking_keys_parsed(rec):
                            ck = (ctry, k)
                            if ck in overflow:
                                continue
                            b = idx_by_country[ctry].get(k)
                            if b is None:
                                idx_by_country[ctry][k] = [j]
                            elif len(b) < 750:
                                b.append(j)
                            else:
                                del idx_by_country[ctry][k]
                                overflow.add(ck)
                if all(n_distractors[c] >= 200000 for c in ("us", "india")) and len(other_by_id) >= len(target_ids_needed) + 400000:
                    break

    log(f"Indexed {len(other_records):,} candidate records ({len(target_ids_needed):,} target pool + {dict(n_distractors)} distractors) with {sum(map(len, idx_by_country.values())):,} country-key buckets.")

    # 4. Load trained model & calibrated config
    model_path = Path("models/lgbm_model.pkl")
    cfg_path = Path("models/calibrated_config.json")
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    clf.set_params(n_jobs=2)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    threshold = cfg.get("threshold", 0.90)
    min_top = cfg.get("min_top", 0.90)
    log(f"Model: {model_path.name} | Rules: threshold={threshold:.2f}, min_top={min_top:.2f}")

    # 5. Run inference
    log(f"Running inference on {len(eval_s1):,} held-out entities...")
    raw_preds = {}
    scored_candidates = {}
    total_pairs = 0
    truth_pair_count = sum(len(truth.get(r.id, set())) for r in eval_s1)
    covered_by_12 = covered_by_16 = covered_by_20 = 0
    entities_with_truth = sum(bool(truth.get(r.id, set())) for r in eval_s1)
    entities_with_any_candidate = entities_with_candidate_12 = 0

    for r in eval_s1:
        # Compare candidate caps using the same real held-out records/index.
        cands20 = query_candidates(r, idx_by_country[eval_country[r.id]], max_candidates=20)
        cands = cands20[:12]
        positives = truth.get(r.id, set())
        entities_with_any_candidate += bool(cands20)
        entities_with_candidate_12 += bool(cands)
        covered_by_12 += len(positives.intersection(other_records[j].id for j in cands))
        covered_by_16 += len(positives.intersection(other_records[j].id for j in cands20[:16]))
        covered_by_20 += len(positives.intersection(other_records[j].id for j in cands20))
        if not cands:
            scored_candidates[r.id] = []
            raw_preds[r.id] = []
            continue
        pairs_feats = [compute_pair_features(r, other_records[j]) for j in cands20]
        total_pairs += len(pairs_feats)
        probs = clf.predict_proba(np.array(pairs_feats, dtype=np.float32))[:, 1]
        
        # Location veto
        rescue_eligible = []
        for idx_p, j in enumerate(cands20):
            feats = pairs_feats[idx_p]
            if feats[22] > 0.5 and feats[11] < 0.90:
                probs[idx_p] = 0.0
            rescue_eligible.append(feats[11] >= 0.90 and (feats[20] > 0.5 or feats[17] >= 0.5))

        scored_candidates[r.id] = [(other_records[j].id, float(p), rescue) for j, p, rescue in zip(cands20, probs, rescue_eligible)]
        cap12_probs = np.array([max(p, threshold) if rescue else p for p, rescue in zip(probs[:len(cands)], rescue_eligible[:len(cands)])])
        best_p = max(cap12_probs, default=0.0)
        if best_p < min_top:
            raw_preds[r.id] = []
        else:
            accepted = [other_records[j].id for j, p in zip(cands, cap12_probs) if p >= threshold]
            raw_preds[r.id] = accepted

    # 6. Compute scores BEFORE 1-to-1 disambiguation
    raw_scores = [entity_f05(set(raw_preds[r.id]), truth.get(r.id, set())) for r in eval_s1]
    raw_macro_f05 = float(np.mean(raw_scores))
    raw_singletons = sum(1 for r in eval_s1 if not raw_preds[r.id])
    log("-" * 70)
    log(f"RAW PRE-DISAMBIGUATION SCORE:")
    log(f"  Macro F_0.5: {raw_macro_f05:.4f}")
    log(f"  Predicted Singletons: {raw_singletons:,} ({raw_singletons/len(eval_s1)*100:.1f}%)")
    log("BLOCKING RECALL BY CANDIDATE CAP (top candidates ranked by key-hit count):")
    log(f"  True target pairs covered: cap12={covered_by_12:,}/{truth_pair_count:,} ({covered_by_12/max(truth_pair_count,1):.4f}), cap16={covered_by_16:,}/{truth_pair_count:,} ({covered_by_16/max(truth_pair_count,1):.4f}), cap20={covered_by_20:,}/{truth_pair_count:,} ({covered_by_20/max(truth_pair_count,1):.4f})")
    log(f"  S1 entities with any candidate: {entities_with_any_candidate:,}/{len(eval_s1):,}; with cap12: {entities_with_candidate_12:,}/{len(eval_s1):,}; true-match entities: {entities_with_truth:,}")
    sweep = []
    for cap in (12, 16, 20):
        for th in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
            for mt in (0.70, 0.80, 0.90, 0.95):
                scores = []
                for r in eval_s1:
                    row = scored_candidates[r.id][:cap]
                    row_probs = [(max(p, th) if rescue else p) for _, p, rescue in row]
                    pred = {row[i][0] for i, p in enumerate(row_probs) if p >= th} if max(row_probs, default=0.0) >= mt else set()
                    scores.append(entity_f05(pred, truth.get(r.id, set())))
                sweep.append((float(np.mean(scores)), cap, th, mt))
    log("TOP REAL-HOLDOUT (cap, threshold, min_top) settings with production address rescue, before 1-to-1 cleanup:")
    for cap in (12, 16, 20):
        score, _, th, mt = max((x for x in sweep if x[1] == cap), key=lambda x: x[0])
        log(f"  Best cap={cap}: F_0.5={score:.4f} threshold={th:.2f} min_top={mt:.2f}")
    for score, cap, th, mt in sorted(sweep, reverse=True)[:5]:
        log(f"  Overall: F_0.5={score:.4f} cap={cap} threshold={th:.2f} min_top={mt:.2f}")

    # 7. Apply 1-to-1 Target Conflict Disambiguation
    target_claims = defaultdict(list)
    for eid, preds in raw_preds.items():
        for tid in preds:
            target_claims[tid].append(eid)

    multi_claims = {tid: s1s for tid, s1s in target_claims.items() if len(s1s) > 1}
    log(f"  Target multi-claims detected: {len(multi_claims):,}")

    assigned_1to1 = {}
    for tid, competitors in multi_claims.items():
        # Keep competitor with highest name match
        t_rec = other_records[other_by_id[tid]] if tid in other_by_id else None
        if not t_rec:
            assigned_1to1[tid] = competitors[0]
            continue
        best_s1, best_s = competitors[0], -1.0
        for s1_id in competitors:
            # find record
            s1_rec = next((x for x in eval_s1 if x.id == s1_id), None)
            if s1_rec:
                s = JaroWinkler.normalized_similarity(s1_rec.name_norm, t_rec.name_norm)
                if s > best_s:
                    best_s = s
                    best_s1 = s1_id
        assigned_1to1[tid] = best_s1

    clean_preds = {}
    for r in eval_s1:
        clean_m = []
        for tid in raw_preds[r.id]:
            if tid in multi_claims:
                if assigned_1to1.get(tid) == r.id:
                    clean_m.append(tid)
            else:
                clean_m.append(tid)
        clean_preds[r.id] = clean_m

    clean_scores = [entity_f05(set(clean_preds[r.id]), truth.get(r.id, set())) for r in eval_s1]
    clean_macro_f05 = float(np.mean(clean_scores))
    clean_singletons = sum(1 for r in eval_s1 if not clean_preds[r.id])
    true_singletons = sum(1 for r in eval_s1 if not truth.get(r.id, set()))

    log("-" * 70)
    log(f"FINAL 1-TO-1 DISAMBIGUATED OFFLINE SCORE:")
    log(f"  Macro F_0.5: {clean_macro_f05:.4f}")
    log(f"  Predicted Singletons: {clean_singletons:,} ({clean_singletons/len(eval_s1)*100:.1f}%)")
    log(f"  True Singletons: {true_singletons:,} ({true_singletons/len(eval_s1)*100:.1f}%)")
    log(f"  Score Improvement from 1-to-1 Disambiguation: +{clean_macro_f05 - raw_macro_f05:.4f}")
    log("=" * 70)

if __name__ == "__main__":
    evaluate_heldout(20000)
