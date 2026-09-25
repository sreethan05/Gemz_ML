"""Shared IO utilities, record container and pipeline configuration."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    # --- blocking ---
    rare_df_max: int = 80          # a name token is a blocking key if its document
                                   # frequency in the S2/S3 corpus is <= this
    max_bucket: int = 150          # buckets larger than this are skipped (too generic)
    min_token_len: int = 3        # ignore very short tokens as keys
    sig_tokens: int = 4           # tokens used in the sorted-signature key
    prefix_len: int = 3            # character prefix key length
    # --- blocking: fuzzy vector layer ---
    vector_topk: int = 50          # top-k TF-IDF cosine neighbours per S1 entity
                                   # added to the key-blocking candidates (0 = off)
    per_entity_topk: int = 0       # optional cap on candidates per entity, kept by
                                   # cosine score (0 = keep everything)
    # --- features ---
    char_ngram_lo: int = 3
    char_ngram_hi: int = 5
    tfidf_max_features: int = 60000
    # --- model ---
    seed: int = 42
    n_estimators: int = 700
    learning_rate: float = 0.05
    num_leaves: int = 63
    # --- matching ---
    threshold: float = 0.65        # pair acceptance threshold
    min_top: float = 0.70          # entity-level: if best candidate score is below
                                   # this, predict singleton (protects F_0.5)
    # --- training ---
    val_fraction: float = 0.15


# --------------------------------------------------------------------------- #
# Record container
# --------------------------------------------------------------------------- #

@dataclass
class Record:
    entity_id: str
    name: str
    address: str
    country: str
    # derived by normalize.build_record
    name_norm: str = ""
    addr_norm: str = ""
    name_tokens: frozenset = field(default_factory=frozenset)
    core_tokens: frozenset = field(default_factory=frozenset)
    addr_tokens: frozenset = field(default_factory=frozenset)
    numbers: tuple = ()
    pins: frozenset = field(default_factory=frozenset)
    metaphone: str = ""
    name_prefix: str = ""


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #

def load_source(path: str | Path) -> list[Record]:
    """Load a *_sourceN.tsv file into a list of Records (raw fields only).

    The derived fields are filled in by normalize.build_record so that the
    featurisation stage never touches raw text.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_filter=False,
    )
    for col in ("entity_id", "business_name", "business_address", "country"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing expected column '{col}'")
    return [
        Record(
            entity_id=str(r.entity_id).strip(),
            name=str(r.business_name).strip(),
            address=str(r.business_address).strip(),
            country=str(r.country).strip(),
        )
        for r in df.itertuples(index=False)
    ]


def load_truth(path: str | Path) -> dict[str, set[str]]:
    """Load train_ground_truth.tsv -> {source1_entity_id: set(matched_ids)}."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_filter=False,
    )
    truth: dict[str, set[str]] = {}
    for r in df.itertuples(index=False):
        s1 = str(r.source1_entity_id).strip()
        raw = str(getattr(r, "matched_entity_ids", "") or "").strip()
        matches = {m.strip() for m in raw.split(",") if m.strip()} if raw else set()
        truth.setdefault(s1, set()).update(matches)
    return truth


def _clean_ids(raw) -> list[str]:
    """Accept a list of IDs or a comma-joined string; dedupe, keep order."""
    if raw is None:
        return []
    if isinstance(raw, str):
        ids = [m.strip() for m in raw.split(",") if m.strip()]
    else:
        ids = [str(m).strip() for m in raw if str(m).strip()]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def write_output_tsv(path: str | Path, s1_ids: list[str], id_lists: dict[str, list[str]],
                    id_column: str = "matched_entity_ids") -> None:
    """Write matching_results.tsv / candidate_pairs.tsv.

    One row per source1 entity (always, even when the list is empty),
    tab separated, comma-joined ID lists, no quoting.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"source1_entity_id\t{id_column}\n")
        for s1 in s1_ids:
            fh.write(f"{s1}\t{','.join(_clean_ids(id_lists.get(s1, [])))}\n")
