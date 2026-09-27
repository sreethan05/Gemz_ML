"""v8 pair features: 35 dims, batched via rapidfuzz cdist for speed.

First 33 dims are value-identical to v4.core.compute_pair_features (same order).
Dims 33/34 add n_g4 / a_g4 (char-4gram Jaccard) — computed but unused in v4.
Empty-side semantics unchanged: all name/addr sims are 0.0 when either side empty.
"""
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz import process as rfp
from rapidfuzz.distance import JaroWinkler, Levenshtein

N_FEATURES_V8 = 35

_SCORERS = (
    fuzz.token_sort_ratio,      # n_sort idx2 / a_sort idx11
    fuzz.token_set_ratio,       # n_set idx3  / a_set idx12
    fuzz.partial_ratio,         # n_part idx4 / a_part idx14
    JaroWinkler.normalized_similarity,  # n_jw idx0 / a_jw idx9
    Levenshtein.normalized_similarity,  # n_lev idx1 / a_lev idx10
)


def _grams(s, out):
    for i in range(len(s) - 3):
        out.add(s[i:i + 4])


def batch_pair_features(q, cand_recs):
    """q: EntityRecord, cand_recs: list[EntityRecord] -> (n,35) float32.

    cand_recs' name_norm/addr_norm are read directly; identical values to the
    per-pair v4 function (rapidfuzz cdist == individual calls).
    """
    n = len(cand_recs)
    out = np.zeros((n, N_FEATURES_V8), dtype=np.float32)
    an, bn = q.name_norm, [c.name_norm for c in cand_recs]
    aa, ba = q.addr_norm, [c.addr_norm for c in cand_recs]
    has_n = bool(an) and any(bn)
    has_a = bool(aa) and any(ba)

    if has_n:
        for col, scorer in ((0, _SCORERS[3]), (1, _SCORERS[4]),
                            (2, _SCORERS[0]), (3, _SCORERS[1]), (4, _SCORERS[2])):
            res = rfp.cdist([an], bn, scorer=scorer, dtype=np.float32,
                            workers=1)[0]
            out[:, col] = res if col in (0, 1) else res / 100.0
        out[:, 5] = out[:, 2]   # n_cos slot = n_sort (as in v4)
    # else stays 0 (covers q empty; per-cand empty handled below)
    if has_a:
        for col, scorer in ((9, _SCORERS[3]), (10, _SCORERS[4]),
                            (11, _SCORERS[0]), (12, _SCORERS[1]), (13, _SCORERS[2])):
            res = rfp.cdist([aa], ba, scorer=scorer, dtype=np.float32,
                            workers=1)[0]
            out[:, col] = res if col in (9, 10) else res / 100.0
        out[:, 14] = out[:, 11]  # a_cos slot = a_sort (as in v4)

    a_core = set(q.core_toks)
    q_gn = set()
    if len(an) >= 4:
        _grams(an, q_gn)
    q_ga = set()
    if len(aa) >= 4:
        _grams(aa, q_ga)

    for i, c in enumerate(cand_recs):
        b_core = set(c.core_toks)
        core_u = len(a_core | b_core)
        if core_u:
            out[i, 6] = len(a_core & b_core) / core_u
        if a_core and b_core:
            out[i, 7] = len(a_core & b_core) / min(len(a_core), len(b_core))
        if an and c.name_norm:
            out[i, 8] = min(len(an), len(c.name_norm)) / max(len(an), len(c.name_norm), 1)
        addr_u = len(q.addr_toks | c.addr_toks)
        if addr_u:
            out[i, 15] = len(q.addr_toks & c.addr_toks) / addr_u
        if aa and c.addr_norm:
            out[i, 16] = min(len(aa), len(c.addr_norm)) / max(len(aa), len(c.addr_norm), 1)
        num_u = len(set(q.numbers) | set(c.numbers))
        if num_u:
            out[i, 17] = len(set(q.numbers) & set(c.numbers)) / num_u
        out[i, 18] = len(q.numbers)
        out[i, 19] = len(c.numbers)
        if q.pins and c.pins:
            out[i, 20] = 1.0 if (q.pins & c.pins) else 0.0
            out[i, 21] = 1.0
            if not (q.pins & c.pins):
                out[i, 22] = 1.0
        if q.numbers and c.numbers and not (set(q.numbers) & set(c.numbers)):
            out[i, 23] = 1.0
        n_s, a_s = out[i, 2], out[i, 11]
        n_e, a_e = out[i, 3], out[i, 12]
        out[i, 24] = n_s * a_s
        out[i, 25] = min(n_e, a_e)
        out[i, 26] = (n_e + a_e) / 2.0
        if out[i, 17]:
            out[i, 27] = n_s * out[i, 17]
        out[i, 28] = a_e * out[i, 0]
        out[i, 29] = 1.0 if c.is_s2 else 0.0
        if q.first_tok and q.first_tok == c.first_tok:
            out[i, 30] = 1.0
        if an and an == c.name_norm:
            out[i, 31] = 1.0
        if a_core and a_core == b_core:
            out[i, 32] = 1.0
        # g4 jaccards (v8 additions)
        if an and c.name_norm and len(c.name_norm) >= 4 and len(an) >= 4:
            cg = set()
            _grams(c.name_norm, cg)
            u = len(q_gn | cg)
            if u:
                out[i, 33] = len(q_gn & cg) / u
        if aa and c.addr_norm and len(c.addr_norm) >= 4 and len(aa) >= 4:
            cg = set()
            _grams(c.addr_norm, cg)
            u = len(q_ga | cg)
            if u:
                out[i, 34] = len(q_ga & cg) / u
    return out
