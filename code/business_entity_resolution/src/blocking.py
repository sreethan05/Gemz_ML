"""Candidate generation (blocking).

Two complementary layers, UNIONED into one candidate set:

A. Key blocking - a (S1, S2/S3) pair becomes a candidate if ANY key matches:

  ("t", token)        rare core name tokens (country-agnostic - a rare token is
                      specific enough on its own; df counted over the S2/S3 corpus)
  ("n", number)       address house/street numbers (country-agnostic, capped)
  ("s", ctry, sig)    sorted-token signature of the name + country
  ("p", ctry, prefix) character prefix of the normalised name + country
  ("m", ctry, mph)     metaphone of the longest core name token + country

  Buckets larger than cfg.max_bucket postings are dropped (generic keys),
  which keeps the candidate set linear-ish in corpus size.

B. Vector (fuzzy) blocking - for every S1 record, the top-k most similar
  S2/S3 records by char n-gram TF-IDF cosine over name+address. This layer
  catches pairs that no exact key survives (heavy typos, transliterations,
  reordering). Implemented with chunked sparse matrix products so it scales
  to large corpora without densifying the whole similarity matrix.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .common import Config, Record


def _record_keys(rec: Record, df_counts: dict[str, int], cfg: Config) -> list[tuple]:
    keys: list[tuple] = []

    # 1. rare name tokens (country-agnostic)
    for tok in rec.core_tokens:
        if len(tok) >= cfg.min_token_len and df_counts.get(tok, 0) <= cfg.rare_df_max:
            keys.append(("t", tok))

    # 2. address numeric tokens (country-agnostic)
    for num in rec.numbers:
        keys.append(("n", num))

    # 3. sorted-token signature + country
    sig = " ".join(sorted(rec.core_tokens)[: cfg.sig_tokens])
    if sig:
        keys.append(("s", rec.country, sig))

    # 4. name prefix + country
    if rec.name_prefix:
        keys.append(("p", rec.country, rec.name_prefix))

    # 5. metaphone of the most distinctive (longest) core token + country
    if rec.metaphone:
        keys.append(("m", rec.country, rec.metaphone))

    return keys


def _doc_freq(records: list[Record]) -> dict[str, int]:
    df: dict[str, int] = defaultdict(int)
    for rec in records:
        for tok in rec.core_tokens:
            df[tok] += 1
    return dict(df)


def generate_candidates(
    s1_records: list[Record],
    other_records: list[Record],
    cfg: Config,
) -> tuple[dict[int, set[int]], dict[str, list[int]]]:
    """Return ({s1_index: {other_index, ...}}, {s1_entity_id: [other_entity_id, ...]}).

    Also returns the id-level view used to write candidate_pairs.tsv, so the
    file is exactly what the matching stage scores.
    """
    df_counts = _doc_freq(other_records)

    # ---- build the key -> posting list index over S2/S3 -------------------
    index: dict[tuple, list[int]] = defaultdict(list)
    for j, rec in enumerate(other_records):
        for key in _record_keys(rec, df_counts, cfg):
            index[key].append(j)
    # drop over-sized (generic) buckets
    index = {k: v for k, v in index.items() if len(v) <= cfg.max_bucket}

    # ---- probe with every S1 record ---------------------------------------
    cands_idx: dict[int, set[int]] = {}
    cands_id: dict[str, list[int]] = {}
    for i, rec in enumerate(s1_records):
        hits: set[int] = set()
        for key in _record_keys(rec, df_counts, cfg):
            bucket = index.get(key)
            if bucket:
                hits.update(bucket)
        cands_idx[i] = hits
        cands_id[rec.entity_id] = sorted(other_records[j].entity_id for j in hits)

    return cands_idx, cands_id


def blocking_recall(cands_idx: dict[int, set[int]],
                    s1_records: list[Record],
                    other_by_id: dict[str, int],
                    truth: dict[str, set[str]],
                    entity_ids: set[str] | None = None) -> tuple[float, int, int]:
    """Fraction of ground-truth pairs recovered by blocking (recall ceiling).

    entity_ids: restrict scoring to these S1 entities (used for the validation
    slice); None scores every entity in `truth`-annotated s1_records."""
    found, total = 0, 0
    for i, rec in enumerate(s1_records):
        if entity_ids is not None and rec.entity_id not in entity_ids:
            continue
        matches = truth.get(rec.entity_id)
        if not matches:
            continue
        got = cands_idx.get(i, set())
        for mid in matches:
            total += 1
            j = other_by_id.get(mid)
            if j is not None and j in got:
                found += 1
    return (found / total if total else 1.0), found, total


# --------------------------------------------------------------------------- #
# B. Vector (fuzzy) blocking layer
# --------------------------------------------------------------------------- #

class VectorIndex:
    """Char n-gram TF-IDF cosine index over name+address for fuzzy blocking.

    topk(i, k)   -> the k most similar S2/S3 record indices for S1 record i
    rank(i, js)  -> cosine scores of S1 record i against candidate indices js
                    (used for optional per-entity top-K trimming)
    """

    def __init__(self, s1_records: list[Record], other_records: list[Record],
                 cfg: Config):
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [f"{r.name_norm} {r.addr_norm}" for r in s1_records] + \
                [f"{r.name_norm} {r.addr_norm}" for r in other_records]
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(cfg.char_ngram_lo, cfg.char_ngram_hi),
            min_df=2,
            max_features=cfg.tfidf_max_features,
            dtype=np.float32,
        )
        M = vec.fit_transform(texts or [" "])
        self.M1 = M[: len(s1_records)].tocsr()
        self.M2 = M[len(s1_records):].tocsr()
        self.n_other = len(other_records)
        # keep chunk x n_other <= ~32M cells (~128 MB float32)
        self.chunk = int(min(512, max(1, 32_000_000 // max(1, self.n_other))))

    def topk(self, i: int, k: int) -> set[int]:
        if self.n_other == 0:
            return set()
        sims = (self.M1[i] @ self.M2.T).toarray().ravel()
        k = min(k, sims.size)
        idx = np.argpartition(sims, -k)[-k:]
        return {int(j) for j in idx if sims[j] > 1e-6}

    def topk_bulk(self, k: int) -> dict[int, set[int]]:
        """Top-k for every S1 record, chunked to bound memory."""
        out: dict[int, set[int]] = {}
        if self.n_other == 0:
            return out
        for start in range(0, self.M1.shape[0], self.chunk):
            sims = (self.M1[start:start + self.chunk] @ self.M2.T).toarray()
            k = min(k, sims.shape[1])
            idx = np.argpartition(sims, -k, axis=1)[:, -k:]
            for r in range(sims.shape[0]):
                i = start + r
                out[i] = {int(j) for j in idx[r] if sims[r, j] > 1e-6}
        return out

    def trim(self, i: int, cand: set[int], k: int) -> set[int]:
        """Keep only the k best candidates of S1 record i by cosine score."""
        if len(cand) <= k:
            return cand
        js = np.fromiter(cand, dtype=np.int64, count=len(cand))
        prod = self.M1[i] @ self.M2[js].T
        sims = np.asarray(prod.todense()).ravel()
        keep = np.argsort(-sims)[:k]
        return {int(js[j]) for j in keep}
