"""v5_optimized model: Ensemble LightGBM with advanced training"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import Config
from .config import Config as V5Config


def train_single_model(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    seed: int,
) -> lgb.LGBMClassifier:
    """Train a single LightGBM model"""
    pos_w = float(np.sum(y == 0)) / max(float(np.sum(y == 1)), 1.0)
    pos_w = min(pos_w, 20.0)
    
    model = lgb.LGBMClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=1,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        scale_pos_weight=pos_w,
        random_state=seed,
        n_jobs=2,
        verbose=-1,
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_gain_to_split=0.01,
    )
    
    model.fit(X, y)
    return model


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> list[lgb.LGBMClassifier]:
    """Train ensemble of models with different seeds"""
    models = []
    for i, seed in enumerate(cfg.ensemble_seeds):
        print(f"  Training ensemble model {i+1}/{len(cfg.ensemble_seeds)} (seed={seed})...")
        t0 = time.time()
        model = train_single_model(X, y, cfg, seed)
        print(f"    Done in {time.time()-t0:.1f}s")
        models.append(model)
    return models


def predict_probs_ensemble(models: list[Any], X: np.ndarray) -> np.ndarray:
    """Average predictions from ensemble"""
    probs = np.zeros(X.shape[0], dtype=np.float32)
    for model in models:
        if hasattr(model, "predict_proba"):
            probs += model.predict_proba(X)[:, 1]
        else:
            probs += np.clip(model.decision_function(X), 0.0, 1.0)
    return probs / len(models)


def save_ensemble(models: list[Any], cfg: Config, feature_names: list[str], val_scores: dict):
    """Save ensemble models and metadata"""
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    
    for i, model in enumerate(models):
        path = cfg.models_dir / f"lgbm_model_{i}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
    
    # Save config
    meta = {
        "n_estimators": cfg.n_estimators,
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "max_depth": cfg.max_depth,
        "min_child_samples": cfg.min_child_samples,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_lambda": cfg.reg_lambda,
        "ensemble_seeds": list(cfg.ensemble_seeds),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "threshold": cfg.threshold,
        "min_top": cfg.min_top,
        "val_scores": val_scores,
    }
    with open(cfg.models_dir / "ensemble_config.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_ensemble(cfg: Config) -> list[Any]:
    """Load ensemble models"""
    models = []
    for i in range(len(cfg.ensemble_seeds)):
        path = cfg.models_dir / f"lgbm_model_{i}.pkl"
        with open(path, "rb") as f:
            models.append(pickle.load(f))
    return models


def predict_probs(model: Any, X: np.ndarray) -> np.ndarray:
    """Single model prediction"""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.clip(model.decision_function(X), 0.0, 1.0)