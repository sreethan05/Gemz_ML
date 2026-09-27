"""v5_optimized matching: Per-country and per-entity calibrated decision rules"""
from __future__ import annotations

import json
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import Config


def make_predictions(
    cand_scores: dict[str, list[tuple[str, float]]],
    threshold: float,
    min_top: float,
) -> dict[str, list[str]]:
    """Basic decision rule: accept if p >= threshold, singleton if best < min_top"""
    out: dict[str, list[str]] = {}
    for s1, scored in cand_scores.items():
        if not scored:
            out[s1] = []
            continue
        best = max(p for _, p in scored)
        accepted = [oid for oid, p in scored if p >= threshold]
        if accepted and best < min_top:
            accepted = []
        out[s1] = sorted(accepted)
    return out


def make_predictions_adaptive(
    cand_scores: dict[str, list[tuple[str, float]]],
    threshold: float,
    min_top: float,
    country: str,
    s1_country: dict[str, str],
) -> dict[str, list[str]]:
    """Adaptive decision rule with country-specific and entity-specific logic"""
    out: dict[str, list[str]] = {}
    
    # Country-specific adjustments
    country_adj = {
        "us": {"th": 0.0, "mt": 0.0},      # US: standard
        "india": {"th": -0.05, "mt": -0.05},  # India: slightly lower (more noisy)
        "france": {"th": -0.05, "mt": -0.05}, # France: similar to India
    }
    adj = country_adj.get(country.lower(), {"th": 0.0, "mt": 0.0})
    
    th = threshold + adj["th"]
    mt = min_top + adj["mt"]
    
    for s1, scored in cand_scores.items():
        if not scored:
            out[s1] = []
            continue
        
        best = max(p for _, p in scored)
        
        # Entity-specific adjustment: if very few candidates, be more lenient
        if len(scored) <= 2:
            th_eff = max(th - 0.05, 0.1)
            mt_eff = max(mt - 0.05, 0.1)
        elif len(scored) >= 20:
            th_eff = th + 0.05
            mt_eff = mt + 0.05
        else:
            th_eff = th
            mt_eff = mt
        
        accepted = [oid for oid, p in scored if p >= th_eff]
        if accepted and best < mt_eff:
            accepted = []
        out[s1] = sorted(accepted)
    
    return out


def sweep_thresholds(
    cand_scores: dict[str, list[tuple[str, float]]],
    truth: dict[str, set[str]],
    grid=None,
) -> tuple[float, float, float]:
    """Grid-search (threshold, min_top) maximising macro F_0.5"""
    import numpy as np
    
    if grid is None:
        grid = [(round(t, 2), round(mt, 2))
                for t in np.arange(0.20, 0.901, 0.03)
                for mt in {round(t, 2), round(t + 0.05, 2), round(t + 0.10, 2), round(t + 0.15, 2), 0.95}]
    
    def entity_f05(pred: set[str], truth: set[str]) -> float:
        if not pred and not truth:
            return 1.0
        tp = len(pred & truth)
        if tp == 0:
            return 0.0
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(truth) if truth else 0.0
        return (1.25 * precision * recall) / (0.25 * precision + recall)
    
    def _predict_entity(scored: list[tuple[str, float]], t: float, min_top: float) -> set[str]:
        best = max(p for _, p in scored) if scored else 0.0
        if not scored or best < min_top:
            return set()
        return {oid for oid, p in scored if p >= t}
    
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


def sweep_thresholds_per_country(
    cand_scores: dict[str, list[tuple[str, float]]],
    truth: dict[str, set[str]],
    s1_country: dict[str, str],
) -> dict[str, tuple[float, float, float]]:
    """Sweep thresholds per country"""
    countries = set(s1_country.values())
    results = {}
    
    for c in countries:
        c_entities = [e for e in truth.keys() if s1_country.get(e) == c]
        if not c_entities:
            continue
        c_scores = {e: cand_scores.get(e, []) for e in c_entities}
        c_truth = {e: truth[e] for e in c_entities}
        t, mt, score = sweep_thresholds(c_scores, c_truth)
        results[c] = (t, mt, score)
        print(f"  {c}: threshold={t:.2f} min_top={mt:.2f} F0.5={score:.4f}")
    
    return results


def enforce_1to1(
    preds: dict[str, list[str]],
    cand_scores: dict[str, list[tuple[str, float]]],
) -> dict[str, list[str]]:
    """Enforce 1-to-1 constraint: each S2/S3 entity matched to at most one S1"""
    # Build claims
    target_claims: dict[str, list[tuple[str, float]]] = {}
    for s1, scored in cand_scores.items():
        for oid, p in scored:
            if oid in preds.get(s1, []):
                target_claims.setdefault(oid, []).append((s1, p))
    
    # Resolve conflicts
    resolved = {}
    for oid, claims in target_claims.items():
        if len(claims) == 1:
            resolved[oid] = claims[0][0]
        else:
            # Keep highest probability claim
            claims.sort(key=lambda x: -x[1])
            resolved[oid] = claims[0][0]
    
    # Apply
    clean_preds = {}
    for s1, matches in preds.items():
        clean_preds[s1] = [oid for oid in matches if resolved.get(oid) == s1]
    
    return clean_preds


def load_country_rules(config_path: str) -> dict:
    """Load per-country decision rules"""
    with open(config_path) as f:
        return json.load(f)