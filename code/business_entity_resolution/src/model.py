"""Gradient-boosted pair classifier with a graceful sklearn fallback."""
from __future__ import annotations

import numpy as np

from .common import Config


def train_model(X: np.ndarray, y: np.ndarray, cfg: Config):
    """Train a binary pair classifier. Prefers LightGBM (MIT licence); falls
    back to scikit-learn's HistGradientBoosting (BSD-3) if lightgbm cannot be
    imported (both are permissively licensed, << 8B parameters)."""
    try:
        import lightgbm as lgb
        pos_w = float(np.sum(y == 0)) / max(float(np.sum(y == 1)), 1.0)
        model = lgb.LGBMClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            scale_pos_weight=min(pos_w, 20.0),
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            min_child_samples=20,
            reg_lambda=1.0,
            random_state=cfg.seed,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:  # pragma: no cover - fallback path
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.num_leaves,
            random_state=cfg.seed,
        )
    model.fit(X, y)
    return model


def predict_probs(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.clip(model.decision_function(X), 0.0, 1.0)
