"""Text normalisation for business names and addresses.

Design goals:
  * every country is treated as an opaque string label (France is unseen in
    training) -> nothing here is country-specific or hard-coded to {US, India};
  * keep BOTH raw and normalised forms: several similarity features work
    better on raw text, while blocking works on canonical forms.
"""
from __future__ import annotations

import re
import unicodedata

from jellyfish import metaphone as _metaphone

from .common import Record

# --------------------------------------------------------------------------- #
# Token tables
# --------------------------------------------------------------------------- #

# legal / corporate suffixes and filler words stripped from "core" name tokens
LEGAL_TOKENS = {
    # US & International
    "corp", "corporation", "inc", "incorporated", "llc", "llp", "ltd", "limited",
    "co", "company", "cos", "plc", "holdings", "group", "enterprises",
    "enterprise", "services", "solutions", "technologies", "tech",
    "international", "internationl", "intl", "consulting", "associates",
    "trading", "traders", "industries", "agency", "agencies", "stores", "store",
    "and", "the", "of", "for", "usa", "us",
    # India
    "pvt", "private", "sons", "india",
    # France
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scop", "gie",
    "societe", "ets", "etablissements", "cie", "france", "fr",
}

# common words that carry no address signal
ADDR_STOP = {
    "near", "opp", "opposite", "behind", "beside", "adj", "adjacent", "next",
    "to", "the", "and", "at", "post", "po", "no", "room", "floor", "flr",
    "dist", "district", "taluk", "tehsil", "road", "rd", "street", "st",
    "lane", "ln", "avenue", "ave", "highway", "cross", "main", "phase",
    "sector", "sec", "block", "building", "bldg", "house", "flat", "flt",
    "shop", "gala", "complex", "area", "nagar", "colony", "enclave",
    "vihar", "layout", "city", "state", "pin", "code", "zip",
    "india", "us", "usa", "france",
    # France address stops
    "rue", "chemin", "quai", "allee", "route", "cedex", "bp",
}

# address abbreviations expanded to a canonical form (bidirectional noise -> one form)
ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "blv": "boulevard", "bd": "boulevard",
    "ln": "lane", "dr": "drive", "drv": "drive", "hwy": "highway",
    "pkwy": "parkway", "cir": "circle", "ct": "court", "plz": "plaza",
    "pl": "place", "sq": "square", "ter": "terrace", "apt": "apartment",
    "ste": "suite", "bldg": "building", "no": "number", "num": "number",
    "mgr": "marg", "sect": "sector", "sec": "sector", "ph": "phase",
    "gnd": "ground", "flt": "flat", "hse": "house", "soc": "society",
    "xing": "crossing", "chowk": "chowk", "br": "branch",
    # french-flavoured variants fold onto the same canonical words
    "rue": "street", "avenu": "avenue", "chem": "chemin", "quai": "quay",
    "all": "allee", "rte": "route", "imp": "impasse",
}

# keep unicode letters/digits and whitespace, strip punctuation/symbols:
# non-Latin scripts (e.g. Devanagari names in Indian records) must survive
# normalisation, or every similarity feature would collapse to garbage.
_NON_ALNUM = re.compile(r"[^\w\s]+", re.UNICODE)
_NUM_RE = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")


def _to_ascii_fold(s: str) -> str:
    """NFKC-fold + drop combining marks (crude transliteration to ascii)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return unicodedata.normalize("NFKC", s)


def normalize_text(s: str) -> str:
    s = _to_ascii_fold(s or "").lower()
    s = s.replace("&", " and ")
    s = _NON_ALNUM.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def _expand_abbrev(tokens: list[str]) -> list[str]:
    return [ADDR_ABBREV.get(t, t) for t in tokens]


def _numbers_from(tokens: list[str]) -> tuple:
    out = []
    for t in tokens:
        for n in _NUM_RE.findall(t):
            if len(n) >= 2 or not tokens:  # ignore single digits
                out.append(n.lstrip("0") or "0")
    return tuple(sorted(set(out)))


def _pins_from(tokens: list[str]) -> frozenset:
    # standalone 5- or 6-digit tokens (US ZIP / India PIN / France CP)
    return frozenset(t for t in tokens if t.isdigit() and 5 <= len(t) <= 6)


def build_record(entity_id: str, name: str, address: str, country: str) -> Record:
    r = Record(entity_id=entity_id, name=name, address=address,
               country=normalize_text(country) or "unknown")
    name_toks = normalize_text(name).split()
    addr_toks = _expand_abbrev(normalize_text(address).split())

    r.name_tokens = frozenset(name_toks)
    r.core_tokens = frozenset(t for t in name_toks
                              if t not in LEGAL_TOKENS and len(t) >= 2)
    r.addr_tokens = frozenset(t for t in addr_toks if t not in ADDR_STOP and len(t) >= 2)
    r.numbers = _numbers_from(addr_toks)
    r.pins = _pins_from(addr_toks)
    r.name_norm = " ".join(name_toks)
    r.addr_norm = " ".join(addr_toks)
    core_sorted = sorted(r.core_tokens, key=lambda t: (-len(t), t))
    # metaphone of the LONGEST core token: the most distinctive one, and the
    # least likely to be entirely consumed by a typo
    r.metaphone = _metaphone(core_sorted[0]) if core_sorted else ""
    r.name_prefix = r.name_norm[:3] if len(r.name_norm) >= 3 else r.name_norm
    return r


def build_records_from_raw(rows) -> list[Record]:
    """rows: iterable of Record objects with only raw fields filled."""
    return [build_record(r.entity_id, r.name, r.address, r.country) for r in rows]
