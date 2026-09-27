"""v5_optimized: End-to-end pipeline runner"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from .config import Config
from .blocking import generate_candidates, blocking_recall, MultiStrategyBlocker, VectorIndex
from .features import PairFeaturizer
from .matching import (
    make_predictions, 
    make_predictions_adaptive,
    sweep_thresholds,
    sweep_thresholds_per_country,
    enforce_1to1,
)
from .model import train_ensemble, predict_probs_ensemble, save_ensemble, load_ensemble
from common import load_source, load_truth, write_output_tsv
from .blocking import build_record


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_pairs(
    s1_records: list,
    other_records: list,
    other_by_id: dict,
    cands_idx: dict,
    truth: dict,
    entity_ids: set,
) -> tuple[list, list, list]:
    """Build training/validation pairs from blocked candidates + truth"""
    pairs, labels, id_pairs = [], [], []
    for i, rec in enumerate(s1_records):
        if rec.entity_id not in entity_ids:
            continue
        matches = truth.get(rec.entity_id, set())
        got = cands_idx.get(i, set())
        for j in got:
            pairs.append((i, j))
            labels.append(1 if other_records[j].entity_id in matches else 0)
            id_pairs.append((rec.entity_id, other_records[j].entity_id))
        for mid in matches:
            j = other_by_id.get(mid)
            if j is not None and j not in got:
                pairs.append((i, j))
                labels.append(1)
                id_pairs.append((rec.entity_id, mid))
    return pairs, labels, id_pairs


def run_training(cfg: Config):
    """Full training pipeline"""
    log("=" * 60)
    log("v5_optimized TRAINING PIPELINE")
    log("=" * 60)
    
    # 1. Load training data
    log("Loading training data...")
    train = cfg.train_dir
    s1_raw = load_source(train / "train_source1.tsv")
    s2_raw = load_source(train / "train_source2.tsv")
    s3_raw = load_source(train / "train_source3.tsv")
    truth = load_truth(train / "train_ground_truth.tsv")
    
    s1 = [build_record(r.entity_id, r.name, r.address, r.country) for r in s1_raw]
    s2 = [build_record(r.entity_id, r.name, r.address, r.country) for r in s2_raw]
    s3 = [build_record(r.entity_id, r.name, r.address, r.country) for r in s3_raw]
    other = s2 + s3
    other_by_id = {r.entity_id: j for j, r in enumerate(other)}
    
    log(f"Train: |S1|={len(s1)} |S2|={len(s2)} |S3|={len(s3)} Truth entities={len(truth)}")
    
    # 2. Validation split
    rng = random.Random(cfg.seed)
    all_s1_ids = [r.entity_id for r in s1]
    shuffled = all_s1_ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * cfg.val_fraction))
    val_ids = set(shuffled[:n_val])
    fit_ids = set(shuffled[n_val:])
    log(f"Split: {len(fit_ids)} fit / {len(val_ids)} validation")
    
    # 3. Load test data for TF-IDF fitting
    log("Loading test data for TF-IDF fitting...")
    test = cfg.test_dir
    t1_raw = load_source(test / "test_source1.tsv")
    t2_raw = load_source(test / "test_source2.tsv")
    t3_raw = load_source(test / "test_source3.tsv")
    t1 = [build_record(r.entity_id, r.name, r.address, r.country) for r in t1_raw]
    t2 = [build_record(r.entity_id, r.name, r.address, r.country) for r in t2_raw]
    t3 = [build_record(r.entity_id, r.name, r.address, r.country) for r in t3_raw]
    log(f"Test: |S1|={len(t1)} |S2|={len(t2)} |S3|={len(t3)}")
    
    # 4. Blocking on train
    log("Blocking on train corpus (multi-strategy + vector)...")
    cands_idx, _, n_key, n_cand = generate_candidates(s1, other, cfg)
    rec_all, f_all, t_all = blocking_recall(cands_idx, s1, other_by_id, truth)
    rec_val, f_val, t_val = blocking_recall(cands_idx, s1, other_by_id, truth, val_ids)
    log(f"Blocking: {n_key} key pairs + vector -> {n_cand} candidates")
    log(f"  Recall (all): {rec_all:.4f} ({f_all}/{t_all})")
    log(f"  Recall (val): {rec_val:.4f} ({f_val}/{t_val})")
    
    # 5. Featurize
    log("Fitting TF-IDF and featurizing...")
    fe = PairFeaturizer(cfg).fit(s1 + other + t1 + t2 + t3)
    log(f"  Features: {len(fe.feature_names)}")
    
    fit_pairs, fit_labels, _ = build_pairs(s1, other, other_by_id, cands_idx, truth, fit_ids)
    val_pairs, val_labels, val_id_pairs = build_pairs(s1, other, other_by_id, cands_idx, truth, val_ids)
    log(f"Training pairs: {len(fit_pairs)} ({sum(fit_labels)} pos)")
    log(f"Validation pairs: {len(val_pairs)} ({sum(val_labels)} pos)")
    
    X_fit = fe.transform(s1 + other, [(i, len(s1) + j) for i, j in fit_pairs])
    X_val = fe.transform(s1 + other, [(i, len(s1) + j) for i, j in val_pairs])
    
    # 6. Train ensemble
    log("Training ensemble models...")
    models = train_ensemble(X_fit, np.array(fit_labels, dtype=np.int8), cfg)
    
    # 7. Validate and tune thresholds
    log("Tuning decision rules...")
    val_probs = predict_probs_ensemble(models, X_val)
    val_scores: dict[str, list[tuple[str, float]]] = {e: [] for e in val_ids}
    for (s1_id, other_id), p in zip(val_id_pairs, val_probs):
        val_scores[s1_id].append((other_id, float(p)))
    val_truth = {e: truth.get(e, set()) for e in val_ids}
    
    # Per-country threshold tuning
    s1_country = {r.entity_id: r.country for r in s1}
    country_rules = sweep_thresholds_per_country(val_scores, val_truth, s1_country)
    
    # Global best
    t, mt, val_f05 = sweep_thresholds(val_scores, val_truth)
    log(f"Global best: threshold={t:.2f} min_top={mt:.2f} F0.5={val_f05:.4f}")
    
    # 8. Save ensemble and config
    val_scores_dict = {c: {"threshold": r[0], "min_top": r[1], "f05": r[2]} for c, r in country_rules.items()}
    save_ensemble(models, cfg, fe.feature_names, val_scores_dict)
    
    # 9. Test prediction
    log("Blocking on test corpus...")
    test_other = t2 + t3
    t_cands_idx, t_cands_id, t_key, n_tcand = generate_candidates(t1, test_other, cfg)
    log(f"Test blocking: {t_key} key pairs + vector -> {n_tcand} candidates")
    
    t_pairs = [(i, len(t1) + j) for i in range(len(t1)) for j in t_cands_idx.get(i, ())]
    X_test = fe.transform(t1 + test_other, t_pairs)
    
    log("Scoring test candidates...")
    t_probs = predict_probs_ensemble(models, X_test)
    
    t_scores: dict[str, list[tuple[str, float]]] = {r.entity_id: [] for r in t1}
    for (i, j), p in zip(t_pairs, t_probs):
        s1_id = t1[i].entity_id
        t_scores[s1_id].append((test_other[j - len(t1)].entity_id, float(p)))
    
    # 10. Apply decision rules
    log("Applying decision rules...")
    t1_country = {r.entity_id: r.country for r in t1}
    preds = make_predictions_adaptive(t_scores, t, mt, "us", t1_country)
    preds = enforce_1to1(preds, t_scores)
    
    # 11. Output
    t1_ids = [r.entity_id for r in t1]
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    write_output_tsv(cfg.out_dir / "matching_results.tsv", t1_ids, preds)
    write_output_tsv(cfg.out_dir / "candidate_pairs.tsv", t1_ids, t_cands_id, 
                     id_column="candidate_entity_ids")
    
    n_pred = sum(len(v) for v in preds.values())
    n_singletons = sum(1 for v in preds.values() if not v)
    log(f"Predicted {n_pred} matches; {n_singletons}/{len(t1)} singletons")
    log(f"Output written to {cfg.out_dir}")
    
    return val_f05, country_rules


def run_inference(cfg: Config):
    """Run inference only (models already trained)"""
    log("=" * 60)
    log("v5_optimized INFERENCE PIPELINE")
    log("=" * 60)
    
    # Load models
    log("Loading ensemble models...")
    models = load_ensemble(cfg)
    with open(cfg.models_dir / "ensemble_config.json") as f:
        meta = json.load(f)
    
    t = meta.get("threshold", 0.50)
    mt = meta.get("min_top", 0.65)
    feature_names = meta["feature_names"]
    country_rules = meta.get("val_scores", {})
    
    log(f"Loaded {len(models)} models, threshold={t:.2f}, min_top={mt:.2f}")
    
    # Load data
    test = cfg.test_dir
    t1_raw = load_source(test / "test_source1.tsv")
    t2_raw = load_source(test / "test_source2.tsv")
    t3_raw = load_source(test / "test_source3.tsv")
    t1 = [build_record(r.entity_id, r.name, r.address, r.country) for r in t1_raw]
    t2 = [build_record(r.entity_id, r.name, r.address, r.country) for r in t2_raw]
    t3 = [build_record(r.entity_id, r.name, r.address, r.country) for r in t3_raw]
    
    # Need training data for TF-IDF fitting
    train = cfg.train_dir
    s1_raw = load_source(train / "train_source1.tsv")
    s2_raw = load_source(train / "train_source2.tsv")
    s3_raw = load_source(train / "train_source3.tsv")
    s1 = [build_record(r.entity_id, r.name, r.address, r.country) for r in s1_raw]
    s2 = [build_record(r.entity_id, r.name, r.address, r.country) for r in s2_raw]
    s3 = [build_record(r.entity_id, r.name, r.address, r.country) for r in s3_raw]
    
    # Featurize
    log("Fitting TF-IDF...")
    fe = PairFeaturizer(cfg)
    fe.feature_names = feature_names
    fe.name_vec = fe.name_vec.__class__()
    fe.addr_vec = fe.addr_vec.__class__()
    # We need to re-fit vectorizers - load from saved or re-fit
    fe.fit(s1 + s2 + s3 + t1 + t2 + t3)
    
    # Blocking
    log("Blocking on test...")
    test_other = t2 + t3
    t_cands_idx, t_cands_id, _, _ = generate_candidates(t1, test_other, cfg)
    
    t_pairs = [(i, len(t1) + j) for i in range(len(t1)) for j in t_cands_idx.get(i, ())]
    X_test = fe.transform(t1 + test_other, t_pairs)
    
    # Score
    log("Scoring...")
    t_probs = predict_probs_ensemble(models, X_test)
    
    t_scores: dict[str, list[tuple[str, float]]] = {r.entity_id: [] for r in t1}
    for (i, j), p in zip(t_pairs, t_probs):
        s1_id = t1[i].entity_id
        t_scores[s1_id].append((test_other[j - len(t1)].entity_id, float(p)))
    
    # Apply rules
    t1_country = {r.entity_id: r.country for r in t1}
    preds = make_predictions_adaptive(t_scores, t, mt, "us", t1_country)
    preds = enforce_1to1(preds, t_scores)
    
    # Output
    t1_ids = [r.entity_id for r in t1]
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    write_output_tsv(cfg.out_dir / "matching_results.tsv", t1_ids, preds)
    write_output_tsv(cfg.out_dir / "candidate_pairs.tsv", t1_ids, t_cands_id,
                     id_column="candidate_entity_ids")
    
    n_pred = sum(len(v) for v in preds.values())
    n_singletons = sum(1 for v in preds.values() if not v)
    log(f"Predicted {n_pred} matches; {n_singletons}/{len(t1)} singletons")


def main():
    ap = argparse.ArgumentParser(description="v5_optimized Entity Resolution Pipeline")
    ap.add_argument("--mode", choices=["train", "infer"], default="train")
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--models-dir", default="models/v5_optimized")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--vector-topk", type=int, default=100)
    ap.add_argument("--per-entity-topk", type=int, default=0)
    args = ap.parse_args()
    
    cfg = Config()
    cfg.train_dir = Path(args.train_dir)
    cfg.test_dir = Path(args.test_dir)
    cfg.out_dir = Path(args.out_dir)
    cfg.models_dir = Path(args.models_dir)
    cfg.seed = args.seed
    cfg.val_fraction = args.val_fraction
    cfg.vector_topk = args.vector_topk
    cfg.per_entity_topk = args.per_entity_topk
    
    if args.mode == "train":
        run_training(cfg)
    else:
        run_inference(cfg)


if __name__ == "__main__":
    main()