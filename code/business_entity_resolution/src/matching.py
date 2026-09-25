"""Per-entity decision rule + output assembly.

F_0.5 is precision-heavy and singletons earn full credit, so the rule is:

  * accept a candidate pair iff p >= threshold;
  * if, after filtering, the entity's best surviving score is < min_top
    (min_top >= threshold), predict singleton (empty list) - this protects
    true singletons, where ANY prediction scores 0.0.
"""
from __future__ import annotations


def make_predictions(cand_scores: dict[str, list[tuple[str, float]]],
                     threshold: float,
                     min_top: float) -> dict[str, list[str]]:
    """cand_scores: {s1_entity_id: [(other_entity_id, p), ...]}."""
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
