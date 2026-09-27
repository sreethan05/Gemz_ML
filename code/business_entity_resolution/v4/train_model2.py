"""Train diverse Model 2 (ExtraTrees-style LightGBM) on the current 39-feature pairs.
Runs in parallel with the chain's eval phase; output = lgbm_model_v6_m2.pkl
(a v6 ensemble member — averaged with the deep model afterwards).
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from v4.core import ensure_hashseed

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "scratch" / "v4"
MODELS = ROOT / "models"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ensure_hashseed()
    Xs, ys = [], []
    for c in ("us", "india"):
        Xs.append(np.load(OUT / f"{c}_train_X.npy"))
        ys.append(np.load(OUT / f"{c}_train_y.npy"))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    del Xs, ys
    log(f"training model2 on {len(X):,} pairs ({int(y.sum()):,} positive)")

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
        n_jobs=2,
    )
    t0 = time.time()
    clf2.fit(X, y)
    log(f"model2 trained in {time.time()-t0:.1f}s")
    with open(MODELS / "lgbm_model_v6_m2.pkl", "wb") as f:
        pickle.dump(clf2, f)
    log("saved -> models/lgbm_model_v6_m2.pkl")


if __name__ == "__main__":
    main()
