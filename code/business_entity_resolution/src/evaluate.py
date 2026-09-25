"""Exact replica of the challenge scoring: per-entity F_0.5, macro-averaged
over ALL Source 1 entities (singletons included), plus threshold sweeping."""
from __future__ import annotations

import numpy as np


def entity_f05(pred: set[str], truth: set[str]) -> float:
    """F_beta (beta=0.5) for one entity, with the challenge's conventions."""
    if not pred and not truth:
        return 1.0                      # correctly predicted singleton
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(preds: dict[str, list[str] | set[str]],
              truth: dict[str, set[str]]) -> float:
    if not truth:
        return 0.0
    scores = [entity_f05(set(preds.get(e, [])), t) for e, t in truth.items()]
    return float(np.mean(scores))


def _predict_entity(scored: list[tuple[str, float]], t: float, min_top: float) -> set[str]:
    best = max(p for _, p in scored) if scored else 0.0
    if not scored or best < min_top:
        return set()
    return {oid for oid, p in scored if p >= t}


def sweep_thresholds(cand_scores: dict[str, list[tuple[str, float]]],
                     truth: dict[str, set[str]],
                     grid=None) -> tuple[float, float, float]:
    """Grid-search (threshold, min_top) maximising macro F_0.5 on validation.

    cand_scores keys must cover all entities in truth (empty list for no
    candidates). Returns (best_threshold, best_min_top, best_score).
    """
    if grid is None:
        grid = [(round(t, 2), round(mt, 2))
                for t in np.arange(0.15, 0.901, 0.05)
                for mt in {round(t, 2), round(t + 0.05, 2), round(t + 0.10, 2), 0.95}]
    best = (0.5, 0.6, -1.0)
    entities = list(truth.keys())
    for t, mt in grid:
        if mt < t:
            continue
        scores = [
            entity_f05(_predict_entity(cand_scores.get(e, []), t, mt), truth[e])
            for e in entities
        ]
        score = float(np.mean(scores)) if scores else 0.0
        if score > best[2]:
            best = (t, mt, score)
    return best
