"""Complementary Model 2 Trainer & Ensembler.

Trains a diverse ExtraTrees-based LightGBM classifier on the full 2.68M pairs.
Evaluates Model 1, Model 2, and the Ensemble on the held-out evaluation sets
to verify the exact F0.5 lift.
"""
import gc
import json
import pickle
import sys
import time
from bisect import bisect_left
from pathlib import Path
import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "scratch" / "v4"
MODELS = ROOT / "models"
COUNTRIES = ("us", "india")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def train_model2():
    log("Loading full 2.68M training dataset...")
    Xs, ys = [], []
    for c in COUNTRIES:
        Xs.append(np.load(OUT / f"{c}_train_X.npy"))
        ys.append(np.load(OUT / f"{c}_train_y.npy"))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    del Xs, ys
    gc.collect()

    log(f"Training Diverse Complementary Model 2 on {len(X):,} pairs...")
    clf2 = lgb.LGBMClassifier(
        n_estimators=550,
        learning_rate=0.04,
        num_leaves=96,
        max_depth=9,
        min_child_samples=30,
        subsample=0.80,
        colsample_bytree=0.75,
        extra_trees=True,
        random_state=123,
        n_jobs=4,
    )
    t0 = time.time()
    clf2.fit(X, y)
    log(f"Model 2 trained in {time.time()-t0:.1f}s")
    
    with open(MODELS / "lgbm_model_v4_m2.pkl", "wb") as f:
        pickle.dump(clf2, f)
    log("Saved Model 2 -> models/lgbm_model_v4_m2.pkl")
    return clf2

def evaluate_ensemble(clf2):
    with open(MODELS / "lgbm_model_v4.pkl", "rb") as f:
        clf1 = pickle.load(f)
    
    log("Evaluating Model 1 vs Model 2 vs Ensemble on Held-Out Validation...")
    for c in COUNTRIES:
        feats = np.load(OUT / f"{c}_eval_feats.npy")
        with open(OUT / f"{c}_eval_candids.pkl", "rb") as f:
            cand_ids = pickle.load(f)
        with open(OUT / f"{c}_eval_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        eval_truth = {k: set(v) for k, v in meta["eval_truth"].items()}
        eids = [r[0] for r in meta["eval_recs"]]

        p1 = clf1.predict_proba(feats)[:, 1]
        p2 = clf2.predict_proba(feats)[:, 1]
        p_ens = 0.5 * p1 + 0.5 * p2

        for name, probs in [("Model 1 (Single)", p1), ("Model 2 (ExtraTrees)", p2), ("Ensemble (M1+M2)", p_ens)]:
            pos = 0
            acc = 0.0
            for i, cids in enumerate(cand_ids):
                k = min(len(cids), 12)
                ps = probs[pos:pos+len(cids)][:k]
                tr = eval_truth.get(eids[i], set())
                tot = len(tr)
                matched = [cids[j] for j in range(k) if ps[j] >= 0.65]
                cnt = len(matched)
                tp = sum(1 for m in matched if m in tr)
                if cnt == 0:
                    acc += 1.0 if tot == 0 else 0.0
                elif tp == 0:
                    pass
                else:
                    p = tp / cnt
                    r = tp / tot
                    acc += (1.25 * p * r) / (0.25 * p + r)
                pos += len(cids)
            f05 = acc / len(cand_ids)
            log(f"  [{c}] {name:25s} -> Macro F0.5 = {f05:.4f}")

if __name__ == "__main__":
    m2 = train_model2()
    evaluate_ensemble(m2)
