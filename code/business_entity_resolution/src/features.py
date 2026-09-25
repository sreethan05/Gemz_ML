"""Pair featurisation.

Two feature families per (S1, S2/S3) candidate pair:

  * string-similarity features on normalised name / address
    (Jaro-Winkler, Levenshtein, token-sort, token-set, partial ratio,
     character-n-gram TF-IDF cosine) - all language/country agnostic, so an
     unseen country (France) is handled with no change;
  * structural features: address number overlap, PIN/ZIP equality,
     country equality, name-token containment, length ratios.
"""
from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import Config, Record


class PairFeaturizer:
    """Fits char n-gram TF-IDF spaces for names and addresses, then computes
    the pair feature matrix for a list of index pairs."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        kw = dict(
            analyzer="char_wb",
            ngram_range=(cfg.char_ngram_lo, cfg.char_ngram_hi),
            min_df=2,
            max_features=cfg.tfidf_max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.name_vec = TfidfVectorizer(**kw)
        self.addr_vec = TfidfVectorizer(**kw)

    # ------------------------------------------------------------------ fit
    def fit(self, records: list[Record]) -> "PairFeaturizer":
        self.name_vec.fit([r.name_norm or " " for r in records])
        self.addr_vec.fit([r.addr_norm or " " for r in records])
        return self

    @property
    def feature_names(self) -> list[str]:
        return [
            "name_jw", "name_lev", "name_sort", "name_set", "name_partial",
            "name_tfidf", "name_jaccard", "name_contain", "name_lenratio",
            "addr_jw", "addr_lev", "addr_sort", "addr_set", "addr_partial",
            "addr_tfidf", "addr_jaccard", "addr_lenratio",
            "num_overlap", "num_a_count", "num_b_count",
            "pin_equal", "pin_both",
            "country_eq",
            "prod_sort", "min_set", "mean_set",
            "name_sort_x_num", "addr_set_x_name_jw",
            "src_is_s2", "first_word_match", "exact_name_norm", "sig_equal",
        ]

    # ------------------------------------------------------------- transform
    def transform(self, records: list[Record],
                 pairs: list[tuple[int, int]],
                 log_every: int = 200_000) -> np.ndarray:
        """pairs: list of (a_index, b_index) into `records` (a = S1 side)."""
        import time as _time
        n = len(pairs)
        if n == 0:
            return np.zeros((0, len(self.feature_names)), dtype=np.float32)

        name_M = self.name_vec.transform([r.name_norm or " " for r in records])
        addr_M = self.addr_vec.transform([r.addr_norm or " " for r in records])
        ai = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=n)
        bi = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=n)
        name_cos = np.asarray(name_M[ai].multiply(name_M[bi]).sum(axis=1)).ravel()
        addr_cos = np.asarray(addr_M[ai].multiply(addr_M[bi]).sum(axis=1)).ravel()

        X = np.zeros((n, len(self.feature_names)), dtype=np.float32)
        jw = JaroWinkler.normalized_similarity
        lv = Levenshtein.normalized_similarity
        tsr = fuzz.token_sort_ratio
        tset = fuzz.token_set_ratio
        part = fuzz.partial_ratio
        t0 = _time.time()

        for k, (i, j) in enumerate(pairs):
            a, b = records[i], records[j]
            # a missing name (or address) on either side must NOT look like a
            # perfect match: blank-vs-blank similarity scorers return 1.0/100
            has_name = bool(a.name_norm) and bool(b.name_norm)
            has_addr = bool(a.addr_norm) and bool(b.addr_norm)
            an, bn = a.name_norm or " ", b.name_norm or " "
            aa, ba = a.addr_norm or " ", b.addr_norm or " "

            n_jw = jw(an, bn) if has_name else 0.0
            n_lev = lv(an, bn) if has_name else 0.0
            n_sort = tsr(an, bn) / 100.0 if has_name else 0.0
            n_set = tset(an, bn) / 100.0 if has_name else 0.0
            n_part = part(an, bn) / 100.0 if has_name else 0.0

            a_jw = jw(aa, ba) if has_addr else 0.0
            a_lev = lv(aa, ba) if has_addr else 0.0
            a_sort = tsr(aa, ba) / 100.0 if has_addr else 0.0
            a_set = tset(aa, ba) / 100.0 if has_addr else 0.0
            a_part = part(aa, ba) / 100.0 if has_addr else 0.0

            core_u = len(a.core_tokens | b.core_tokens)
            n_jac = len(a.core_tokens & b.core_tokens) / core_u if core_u else 0.0
            if a.core_tokens and b.core_tokens:
                n_cont = (len(a.core_tokens & b.core_tokens)
                          / min(len(a.core_tokens), len(b.core_tokens)))
            else:
                n_cont = 0.0
            n_len = min(len(an), len(bn)) / max(len(an), len(bn), 1)

            addr_u = len(a.addr_tokens | b.addr_tokens)
            a_jac = len(a.addr_tokens & b.addr_tokens) / addr_u if addr_u else 0.0
            a_len = min(len(aa), len(ba)) / max(len(aa), len(ba), 1)

            num_u = len(set(a.numbers) | set(b.numbers))
            num_ov = len(set(a.numbers) & set(b.numbers)) / num_u if num_u else 0.0

            a_first = an.split(" ", 1)[0] if an.strip() else ""
            b_first = bn.split(" ", 1)[0] if bn.strip() else ""

            row = X[k]
            row[0] = n_jw; row[1] = n_lev; row[2] = n_sort; row[3] = n_set
            row[4] = n_part; row[5] = name_cos[k] if has_name else 0.0
            row[6] = n_jac; row[7] = n_cont; row[8] = n_len
            row[9] = a_jw; row[10] = a_lev; row[11] = a_sort; row[12] = a_set
            row[13] = a_part; row[14] = addr_cos[k] if has_addr else 0.0
            row[15] = a_jac; row[16] = a_len
            row[17] = num_ov
            row[18] = len(a.numbers); row[19] = len(b.numbers)
            row[20] = 1.0 if (a.pins and a.pins & b.pins) else 0.0
            row[21] = 1.0 if (a.pins and b.pins) else 0.0
            row[22] = 1.0 if a.country == b.country else 0.0
            row[23] = n_sort * a_sort
            row[24] = min(n_set, a_set)
            row[25] = (n_set + a_set) / 2.0
            row[26] = n_sort * num_ov if num_ov else 0.0
            row[27] = a_set * n_jw
            row[28] = 1.0 if b.entity_id.startswith("S2-") else 0.0
            row[29] = 1.0 if (a_first and a_first == b_first) else 0.0
            row[30] = 1.0 if (has_name and an == bn) else 0.0
            row[31] = 1.0 if (a.core_tokens and a.core_tokens == b.core_tokens) else 0.0
            if log_every and (k + 1) % log_every == 0:
                rate = (k + 1) / max(_time.time() - t0, 1e-9)
                print(f"    featurised {k + 1:,}/{n:,} pairs "
                      f"({rate:,.0f} pairs/s)", flush=True)
        return X
