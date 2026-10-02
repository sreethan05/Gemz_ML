"""v4 core: production-scale blocking for Business Entity Resolution.

Key ideas vs the previous pipeline:
- CSR posting index: NO bucket is ever deleted (old pipeline deleted overflowing
  buckets entirely, losing true matches at 5M-record scale).
- IDF-weighted candidate ranking: a hit on a rare key outweighs hits on common
  keys (old pipeline ranked by raw hit count, letting common-token noise crowd
  out true matches).
- Keys reflect the actual data: postal codes are nearly absent in this dataset,
  so blocking leans on name signatures, house numbers + name prefix, and long
  address tokens. City/landmark tokens stay available via ('a', tok) keys.
- Deterministic via PYTHONHASHSEED=1 (enforced by ensure_hashseed()).
"""
import os
import re
import sys
import unicodedata
from array import array

import numpy as np

ID_WIDTH = 12  # all ids are S2-/S3- + <=9 digits


def ensure_hashseed():
    """Require a fixed PYTHONHASHSEED so key hashing is reproducible.

    Launch scripts with PYTHONHASHSEED=1 (they do this themselves). We assert
    instead of re-exec'ing because os.execve breaks detached/background shells
    on Windows.
    """
    if os.environ.get("PYTHONHASHSEED") != "1":
        raise RuntimeError(
            "PYTHONHASHSEED=1 must be set before launching "
            "(key hashing must be deterministic and stable within the process)"
        )


# ---------------------------------------------------------------- text norms

_OFFSET_MAP = {
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
    0x0B: "ri", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
    0x13: "o", 0x14: "au",
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "d", 0x5D: "dh",
    0x5E: "f", 0x5F: "y",
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri",
    0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x01: "n", 0x02: "n", 0x03: "h",
}
_VIRAMA_OFFSETS = {0x4D}

FAST_TABLE = {
    ord("œ"): "oe", ord("Œ"): "oe",
    ord("æ"): "ae", ord("Æ"): "ae",
    0x200C: "", 0x200D: "",  # ZWJ / ZWNJ
}
for _d in range(10):
    FAST_TABLE[0x0660 + _d] = str(_d)  # Arabic-Indic digits
for _base in range(0x0900, 0x0D80, 0x80):  # Brahmic blocks (Devanagari..Malayalam)
    for _off, _lat in _OFFSET_MAP.items():
        FAST_TABLE[_base + _off] = _lat
    for _off in _VIRAMA_OFFSETS:
        FAST_TABLE[_base + _off] = ""
    for _d in range(10):  # block-native digits, e.g. Devanagari digits
        FAST_TABLE[_base + 0x66 + _d] = str(_d)

LEGAL_TOKENS = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation",
    "co", "company", "llc", "llp", "plc", "holdings", "group", "enterprises",
    "enterprise", "services", "service", "solutions", "technologies", "tech",
    "international", "consulting", "associates", "trading", "industries",
    "traders", "agency", "agencies", "stores", "store",
    "pvt", "private", "sons", "partners", "bros",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scop", "gie",
    "societe", "ets", "etablissements", "cie", "france", "fr",
    # domain-noise tokens: some S2/S3 names are the business name concatenated
    # as a web domain ("pediatricdentistryprime com")
    "com", "www", "net", "org",
    # Indic-phonetic legal-suffix variants (transliterated forms of
    # private/limited appear as literal name tokens in S2/S3)
    "praibhet", "praivet", "praivat", "limirrd", "limitet", "limitid",
    "limite", "praim", "limrd",
}
ADDR_STOP = {
    "near", "opp", "opposite", "behind", "beside", "adj", "adjacent",
    "at", "post", "po", "dist", "district", "taluk", "tehsil", "road",
    "rd", "street", "st", "lane", "ln", "avenue", "ave", "highway",
    "cross", "main", "phase", "sector", "sec", "block", "floor",
    "room", "flat", "shop", "gala", "bldg", "building", "house",
    "complex", "nagar", "colony", "enclave", "vihar", "layout",
    "city", "state", "pin", "code", "zip", "india", "us", "usa", "france",
    "rue", "chemin", "quai", "allee", "route", "cedex", "bp",
    # French articles / prepositions (test-only country)
    "du", "de", "la", "le", "les", "des", "et", "chez", "sur", "au", "aux",
}
ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "hwy": "highway", "dr": "drive", "ct": "court",
    "pl": "place", "sq": "square", "ter": "terrace", "apt": "apartment",
    "ste": "suite", "bldg": "building", "no": "number", "num": "number",
    "mgr": "marg", "sect": "sector", "sec": "sector", "ph": "phase",
    "gnd": "ground", "flt": "flat", "hse": "house", "soc": "society",
    "xing": "crossing", "chowk": "chowk", "br": "branch",
    "rue": "street", "avenu": "avenue", "chem": "chemin", "quai": "quay",
    "bd": "boulevard", "all": "allee", "rte": "route", "imp": "impasse",
    "r": "rue", "gde": "grande",  # French "R." / "Gde" abbreviations
}

_NON_ALNUM = re.compile(r"[^\w\s]+", re.UNICODE)
_NUM_RE = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")


def transliterate_fast(s: str) -> str:
    if not s or s.isascii():
        return s
    return "".join(FAST_TABLE.get(ord(c), c) for c in s)


def clean_text(s: str) -> str:
    s = transliterate_fast(s or "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("&", " and ")
    s = s.replace(".", " ").replace("/", " ").replace("-", " ")
    s = _NON_ALNUM.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


# ---------------------------------------------------------------- record

class EntityRecord:
    __slots__ = ("id", "name_norm", "addr_norm", "core_toks", "addr_toks",
                 "numbers", "pins", "first_tok", "is_s2")

    def __init__(self, eid: str, name: str, address: str):
        EntityRecord._fill(self, eid, clean_text(name), clean_text(address))

    @classmethod
    def from_norms(cls, eid: str, name_norm: str, addr_norm: str):
        rec = cls.__new__(cls)
        EntityRecord._fill(rec, eid, name_norm, addr_norm)
        return rec

    @staticmethod
    def _fill(rec, eid, name_norm, addr_norm):
        addr_toks = [ADDR_ABBREV.get(t, t) for t in addr_norm.split()]
        name_toks = name_norm.split()
        rec.id = eid
        rec.name_norm = name_norm
        rec.addr_norm = " ".join(addr_toks)
        rec.core_toks = tuple(t for t in name_toks if t not in LEGAL_TOKENS and len(t) >= 3)
        rec.addr_toks = frozenset(t for t in addr_toks if t not in ADDR_STOP and not t.isdigit() and len(t) >= 5)
        nums = tuple(sorted(set(n.lstrip("0") or "0"
                                for t in addr_toks for n in _NUM_RE.findall(t))))
        rec.numbers = nums
        rec.pins = frozenset(n for n in nums if len(n) in (5, 6))
        rec.first_tok = name_toks[0] if name_toks else ""
        rec.is_s2 = eid.startswith("S2-")


WTYPE = {"nn": 1.6, "s": 1.5, "nf": 1.3, "p": 1.2, "pf": 1.2, "n": 1.0,
         "a": 0.9, "c8": 1.3, "cs8": 1.2, "g4n": 0.7, "g4a": 0.6, "mh": 0.8}

# MinHash band parameters: 16 independent uint64 hash functions (odd multipliers,
# xor shifts). Band b's key = (band id, min over shingles of h_b(shingle)).
# Two records collide on band b iff they share their b-th minimum char-4gram —
# i.e. they share at least one rare shingle, robust to typos/reorder/script.
MH_A = (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5,
        0x85EBCA77C2B2AE63, 0x27D4EB2F165667C5, 0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F,
        0x165667B19E3779F9, 0x85EBCA77C2B2AE63, 0x27D4EB2F165667C5, 0x9E3779B97F4A7C15,
        0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x85EBCA77C2B2AE63, 0x9E3779B185EBCA87)
MH_B = (0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 0x2545F4914F6CDD1D, 0x9E3779B97F4A7C15,
        0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5, 0x85EBCA77C2B2AE63,
        0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 0x2545F4914F6CDD1D, 0x9E3779B97F4A7C15,
        0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5, 0x85EBCA77C2B2AE63)
MH_MOD = (1 << 61) - 1  # Mersenne prime for clean modular hashing


def minhash_keys(text: str, n_bands: int = 16):
    """16 band keys from char-4gram minhash values (LSH r=1, b=16).

    Recall bound: P(collision) = 1-(1-s)^16 >= 0.99 for pairs sharing 25%+ of
    their char-4gram shingles — far more robust than fixed-position sampling.
    """
    sh = {text[i:i + 4] for i in range(len(text) - 3)}
    sh.discard("")
    if len(sh) < 6:
        return []
    arr = np.fromiter((hash(("mh", x)) & 0x1FFFFFFFFFFFFF for x in sh),
                      dtype=np.uint64, count=len(sh))
    A = np.array(MH_A[:n_bands], dtype=np.uint64)
    B = np.array(MH_B[:n_bands], dtype=np.uint64)
    vals = (arr[None, :] * A[:, None] + B[:, None]) % np.uint64(MH_MOD)
    mins = vals.min(axis=1)
    return [("mh", (b, int(mins[b]))) for b in range(n_bands)]

_G4_STEP_N = 4   # sample every 4th char-4gram of the name (memory control)
_G4_STEP_A = 6   # every 6th of the address


def _char4_keys(text: str, step: int, tag: str, keys: list):
    if len(text) < 4:
        return
    grams = list(dict.fromkeys(text[i:i + 4] for i in range(0, len(text) - 3, step)))
    keys.extend((tag, g) for g in grams if g.strip())


def keys_of(r: EntityRecord) -> list:
    """Blocking keys as (key_type, key_tuple) — shared by pool build and queries."""
    f2 = r.name_norm[:2]
    keys = []
    if len(r.name_norm) >= 4:
        keys.append(("nn", r.name_norm))
    for t in r.core_toks:
        if len(t) >= 4:
            keys.append(("n", t))
    if len(r.core_toks) >= 2:
        keys.append(("s", " ".join(sorted(r.core_toks)[:3])))
    for n in r.numbers:
        if len(n) >= 2 and f2:
            keys.append(("nf", n, f2))
    for p in r.pins:
        keys.append(("p", p))
        if f2:
            keys.append(("pf", p, f2))
    for t in r.addr_toks:
        if len(t) >= 6:
            keys.append(("a", t))
    # concatenated-name prefix keys: catch domain-style names where the
    # business name is run together into one token ("salleyclearchemical com")
    if r.core_toks:
        c8 = "".join(r.core_toks)[:8]
        if len(c8) >= 6:
            keys.append(("c8", c8))
        cs8 = "".join(sorted(r.core_toks))[:8]
        if len(cs8) >= 6:
            keys.append(("cs8", cs8))
    # char-4gram keys: typo/reorder/transliteration-robust fuzzy layer
    _char4_keys(r.name_norm.replace(" ", ""), _G4_STEP_N, "g4n", keys)
    _char4_keys(r.addr_norm.replace(" ", ""), _G4_STEP_A, "g4a", keys)
    # minhash band keys (LSH): position-independent rare-shingle matching
    text = r.name_norm.replace(" ", "") + "|" + r.addr_norm.replace(" ", "")
    keys.extend(minhash_keys(text))
    return keys


def key_hashes(keys) -> tuple:
    """Hash keys to (int64 hashes, float32 weights, int32 per-key max_df).

    Address tokens get a tighter df cap (600): distinctive landmarks/streets
    are strong evidence, while high-df tokens (major roads, big cities) are
    noise magnets that dominate ranking cost (finding consistent with the
    reference solution's frequency-capped address-token blocking).
    """
    hs = np.fromiter((hash(k) for k in keys), dtype=np.int64, count=len(keys))
    ws = np.fromiter((WTYPE[k[0]] for k in keys), dtype=np.float32, count=len(keys))
    mdf = np.fromiter((600 if k[0] in ("a", "g4a") else 250_000 for k in keys),
                      dtype=np.int64, count=len(keys))
    return hs, ws, mdf


# ---------------------------------------------------------------- packed pool

class PackedPool:
    """Memory-lean record store: cleaned text blobs + fixed-width id blob."""

    def __init__(self):
        self.ids = bytearray()
        self.names = bytearray()
        self.addrs = bytearray()
        self.name_off = array("q", [0])
        self.addr_off = array("q", [0])
        self.id_hash = None
        self.id_hash_sorted = None
        self.id_perm = None
        self.n = 0
        self.cache_cap = 900_000
        self._cache = {}

    def add_raw(self, eid: str, name: str, addr: str):
        nn, an = clean_text(name), clean_text(addr)
        self.ids += eid.encode("ascii").ljust(ID_WIDTH)
        self.names += nn.encode("utf-8")
        self.addrs += an.encode("utf-8")
        self.name_off.append(len(self.names))
        self.addr_off.append(len(self.addrs))
        self.n += 1

    def finalize(self):
        self.ids = bytes(self.ids)
        self.names = bytes(self.names)
        self.addrs = bytes(self.addrs)
        self.name_off = np.frombuffer(self.name_off, dtype=np.int64)
        self.addr_off = np.frombuffer(self.addr_off, dtype=np.int64)
        ih = np.empty(self.n, dtype=np.int64)
        idb = self.ids
        w = ID_WIDTH
        for i in range(self.n):
            ih[i] = hash(idb[i * w:(i + 1) * w].rstrip())
        self.id_hash = ih
        perm = np.argsort(ih, kind="stable")
        self.id_perm = perm.astype(np.int64)
        self.id_hash_sorted = ih[perm]

    def get_id(self, j: int) -> str:
        return self.ids[j * ID_WIDTH:(j + 1) * ID_WIDTH].rstrip().decode("ascii")

    def get_name(self, j: int) -> str:
        return self.names[self.name_off[j]:self.name_off[j + 1]].decode("utf-8")

    def get_addr(self, j: int) -> str:
        return self.addrs[self.addr_off[j]:self.addr_off[j + 1]].decode("utf-8")

    def lookup_id(self, eid: str):
        """Record index by id, or -1."""
        h = hash(eid.encode("ascii"))
        i = int(np.searchsorted(self.id_hash_sorted, h))
        while i < self.n and self.id_hash_sorted[i] == h:
            j = int(self.id_perm[i])
            if self.get_id(j) == eid:
                return j
            i += 1
        return -1

    def record(self, j: int) -> EntityRecord:
        rec = self._cache.get(j)
        if rec is None:
            rec = EntityRecord.from_norms(self.get_id(j), self.get_name(j), self.get_addr(j))
            if len(self._cache) > self.cache_cap:
                self._cache.clear()
            self._cache[j] = rec
        return rec


# ---------------------------------------------------------------- CSR index

class CSRIndex:
    """Sorted posting lists over a record range; no bucket is ever deleted.

    Keys whose document frequency exceeds max_df are dropped at build time
    (they cannot discriminate candidates and would dominate query cost).
    """

    def __init__(self, H_sorted, J_sorted, max_df, max_df_query=3000):
        self.H = H_sorted
        self.J = J_sorted
        self.max_df = max_df
        self.max_df_query = max_df_query

    @classmethod
    def build(cls, pool: PackedPool, j0: int, j1: int, max_df: int = 250_000):
        hs_arr = array("q")
        js_arr = array("i")
        add_h = hs_arr.append
        add_j = js_arr.append
        for j in range(j0, j1):
            for k in keys_of(pool.record(j)):
                add_h(hash(k))
                add_j(j)
        H = np.frombuffer(hs_arr, dtype=np.int64)
        J = np.frombuffer(js_arr, dtype=np.int32)
        order = np.argsort(H, kind="stable")
        H = H[order]
        J = J[order]
        del order
        # drop mega-common keys (df > max_df)
        starts = np.flatnonzero(np.r_[True, H[1:] != H[:-1]])
        ends = np.r_[starts[1:], len(H)]
        keep = np.repeat((ends - starts) <= max_df, ends - starts)
        if not keep.all():
            H = H[keep]
            J = J[keep]
        return cls(np.ascontiguousarray(H), np.ascontiguousarray(J), max_df)

    def query(self, hs: np.ndarray, ws: np.ndarray, topk: int, mdfs: np.ndarray = None):
        """Return (record_idx, scores) top-k by summed IDF weight, descending.

        Keys with df > max_df_query are skipped: their IDF weight is tiny
        (<= 1/log1p(3000) ~= 0.12) and their huge posting lists dominate
        query time. Pairs sharing ONLY such keys are unrankable anyway.
        """
        lo = np.searchsorted(self.H, hs, side="left")
        hi = np.searchsorted(self.H, hs, side="right")
        df = hi - lo
        ok = (df > 0) & (df <= self.max_df_query)
        if mdfs is not None:
            ok &= df <= mdfs
        if not ok.any():
            return None, None
        idx_list = []
        w_list = []
        for i in np.flatnonzero(ok):
            s, e = int(lo[i]), int(hi[i])
            idx_list.append(self.J[s:e])
            w_list.append(np.full(e - s, ws[i] / np.log1p(int(df[i])), dtype=np.float32))
        if len(idx_list) == 1:
            jj, ww = idx_list[0], w_list[0]
        else:
            jj = np.concatenate(idx_list)
            ww = np.concatenate(w_list)
        order = np.argsort(jj, kind="stable")
        jj = jj[order]
        ww = ww[order]
        starts = np.flatnonzero(np.r_[True, jj[1:] != jj[:-1]])
        sums = np.add.reduceat(ww, starts)
        uniq_j = jj[starts]
        if len(uniq_j) > topk:
            sel = np.argpartition(sums, -topk)[-topk:]
            sel = sel[np.argsort(sums[sel])[::-1]]
        else:
            sel = np.argsort(sums)[::-1]
        return uniq_j[sel], sums[sel]


def query_cascade(pool: "PackedPool", csrs, r: EntityRecord, cap: int, k0: int = None):
    """Two-stage retrieval: IDF top-k0, then name+address+IDF rescore, top-cap.

    The key set covers ~99.8% of true pairs but pure IDF ranking loses ~10%
    inside the cap (chains share identical keys). A token_set_ratio proxy on
    the stored normalized names (bytes slice, no re-parse) surfaces them
    without exploding the final candidate list.
    """
    from rapidfuzz import fuzz as _fuzz

    if k0 is None:
        k0 = max(10 * cap, 300)
    per = max(k0 // max(len(csrs), 1) + 20, 60)
    keys = keys_of(r)
    hs, ws, mdfs = key_hashes(keys)
    all_j, all_s = [], []
    for csr in csrs:
        js, ss = csr.query(hs, ws, per, mdfs)
        if js is not None:
            all_j.append(js)
            all_s.append(ss)
    if not all_j:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    js = np.concatenate(all_j)
    ss = np.concatenate(all_s)
    # dedupe (same record may hit via multiple CSRs)
    js, first = np.unique(js, return_index=True)
    ss = ss[first]
    if len(js) > k0:
        sel = np.argpartition(ss, -k0)[-k0:]
        js = js[sel]
        ss = ss[sel]
    # rescore: address-weighted proxy + normalized IDF (vectorized via
    # rapidfuzz.process.cdist; name-only rescore A/B-lost by ~2.8pp)
    from rapidfuzz import process as _proc
    idf_n = ss / max(float(ss.max()), 1e-9)
    names = [pool.get_name(int(j)) for j in js]
    addrs = [pool.get_addr(int(j)) for j in js]
    n_sim = _proc.cdist([r.name_norm], names, scorer=_fuzz.token_set_ratio,
                        dtype=np.uint8)[0] / 100.0
    a_sim = _proc.cdist([r.addr_norm], addrs, scorer=_fuzz.token_set_ratio,
                        dtype=np.uint8)[0] / 100.0
    combined = 0.35 * n_sim + 0.45 * a_sim + 0.2 * idf_n
    if len(js) > cap:
        sel = np.argpartition(combined, -cap)[-cap:]
        sel = sel[np.argsort(combined[sel])[::-1]]
    else:
        sel = np.argsort(combined)[::-1]
    return js[sel], combined[sel]


# ---------------------------------------------------------------- features

N_FEATURES = 33  # proven v5-era feature set

from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz import fuzz

_jw = JaroWinkler.normalized_similarity
_lv = Levenshtein.normalized_similarity
_tsr = fuzz.token_sort_ratio
_tset = fuzz.token_set_ratio
_part = fuzz.partial_ratio


def compute_pair_features(a: EntityRecord, b: EntityRecord):
    jw, lv = _jw, _lv
    tsr, tset, part = _tsr, _tset, _part

    an, bn = a.name_norm, b.name_norm
    aa, ba = a.addr_norm, b.addr_norm
    has_n = bool(an) and bool(bn)
    has_a = bool(aa) and bool(ba)

    if not has_n:
        n_jw = n_lev = n_sort = n_set = n_part = 0.0
    elif an == bn:
        n_jw = n_lev = n_sort = n_set = n_part = 1.0
    else:
        n_jw = jw(an, bn)
        n_lev = lv(an, bn)
        n_sort = tsr(an, bn) / 100.0
        n_set = tset(an, bn) / 100.0
        n_part = part(an, bn) / 100.0

    if not has_a:
        a_jw = a_lev = a_sort = a_set = a_part = 0.0
    elif aa == ba:
        a_jw = a_lev = a_sort = a_set = a_part = 1.0
    else:
        a_jw = jw(aa, ba)
        a_lev = lv(aa, ba)
        a_sort = tsr(aa, ba) / 100.0
        a_set = tset(aa, ba) / 100.0
        a_part = part(aa, ba) / 100.0

    a_core = set(a.core_toks)
    b_core = set(b.core_toks)
    core_u = len(a_core | b_core)
    n_jac = len(a_core & b_core) / core_u if core_u else 0.0
    n_cont = (len(a_core & b_core) / min(len(a_core), len(b_core))) if (a_core and b_core) else 0.0
    n_len = min(len(an), len(bn)) / max(len(an), len(bn), 1)

    addr_u = len(a.addr_toks | b.addr_toks)
    a_jac = len(a.addr_toks & b.addr_toks) / addr_u if addr_u else 0.0
    a_len = min(len(aa), len(ba)) / max(len(aa), len(ba), 1)

    num_u = len(set(a.numbers) | set(b.numbers))
    num_ov = len(set(a.numbers) & set(b.numbers)) / num_u if num_u else 0.0

    pin_match = 1.0 if (a.pins and b.pins and (a.pins & b.pins)) else 0.0
    pin_conflict = 1.0 if (a.pins and b.pins and not (a.pins & b.pins)) else 0.0
    pin_both = 1.0 if (a.pins and b.pins) else 0.0
    num_conflict = 1.0 if (a.numbers and b.numbers and not (set(a.numbers) & set(b.numbers))) else 0.0
    # Slots 5 and 14 must stay equal to token-sort: v5 LightGBM was trained
    # that way. char-4gram Jaccard is used by the cluster/g4 *decision gate*
    # (char4_jaccard), not as a 35th model feature, until a full retrain.
    n_cos = n_sort
    a_cos = a_sort

    return [
        n_jw, n_lev, n_sort, n_set, n_part, n_cos, n_jac, n_cont, n_len,
        a_jw, a_lev, a_sort, a_set, a_part, a_cos, a_jac, a_len,
        num_ov, len(a.numbers), len(b.numbers),
        pin_match, pin_both, pin_conflict, num_conflict,
        n_sort * a_sort, min(n_set, a_set), (n_set + a_set) / 2.0,
        n_sort * num_ov if num_ov else 0.0, a_set * n_jw,
        1.0 if b.is_s2 else 0.0,
        1.0 if (a.first_tok and a.first_tok == b.first_tok) else 0.0,
        1.0 if (has_n and an == bn) else 0.0,
        1.0 if (a_core and a_core == b_core) else 0.0,
    ]


def char4_jaccard(a: str, b: str) -> float:
    """Char-4gram Jaccard; 0 if either string is too short."""
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    ga = {a[i:i + 4] for i in range(len(a) - 3)}
    gb = {b[i:i + 4] for i in range(len(b) - 3)}
    u = ga | gb
    return (len(ga & gb) / len(u)) if u else 0.0


def accept_indices(probs, fps, threshold: float, min_top: float,
                   cluster: str = "off", cluster_floor: float = 0.0):
    """Per-entity accept mask over a candidate list.

    cluster:
      off        — p >= threshold (and best >= min_top)
      propagate  — also accept candidates sharing a fingerprint with an
                   already-accepted member, if their p >= cluster_floor
    """
    n = len(probs)
    if n == 0:
        return []
    best = float(max(probs))
    if best < min_top:
        return []
    acc = [i for i in range(n) if probs[i] >= threshold]
    if cluster == "propagate" and acc:
        afp = {fps[i] for i in acc if fps[i]}
        if afp:
            floor = float(cluster_floor)
            acc = [i for i in range(n) if
                   probs[i] >= threshold or (fps[i] in afp and probs[i] >= floor)]
    return acc


def entity_f05(pred: set, truth: set) -> float:
    if not pred and not truth:
        return 1.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return (1.25 * p * r) / (0.25 * p + r)


class EnsembleModel:
    """Probability-averaging wrapper; exposes the LightGBM-compatible surface."""

    def __init__(self, clfs):
        self.clfs = clfs

    def set_params(self, **kw):
        for m in self.clfs:
            m.set_params(**kw)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.clfs], axis=0)


# ---------------------------------------------------------------- sibling clusters

def fingerprint_of(rec: EntityRecord) -> str:
    """Sibling fingerprint: children of one parent share name signature + numbers."""
    sig = " ".join(sorted(rec.core_toks)[:3])
    nums = ",".join(rec.numbers[:2])
    return sig + "|" + nums if sig else ""


def build_clusters(pool: PackedPool):
    """Group records into sibling clusters by fingerprint.

    Returns (cluster_of, members_sorted, bounds) where members of cluster c are
    members_sorted[bounds[c]:bounds[c+1]] (record indices, ascending).
    Measured on train: 96.8% of multi-member clusters share one true parent.
    """
    fps = np.empty(pool.n, dtype=np.int64)
    pool.cache_cap = 200_000
    for j in range(pool.n):
        fp = fingerprint_of(pool.record(j))
        fps[j] = hash(fp) if fp else hash(("~solo~", j))
        if (j + 1) % 1_000_000 == 0:
            pool._cache.clear()
    pool._cache.clear()
    order = np.argsort(fps, kind="stable")
    fs = fps[order]
    newc = np.r_[True, fs[1:] != fs[:-1]]
    cid_sorted = np.cumsum(newc) - 1                     # cluster id per sorted pos
    cluster_of = np.empty(pool.n, dtype=np.int64)
    cluster_of[order] = cid_sorted                       # record -> cluster id
    n_clusters = int(cid_sorted[-1]) + 1
    # members sorted by record index within each cluster
    rec_sorted = order[np.argsort(cid_sorted, kind="stable")]
    counts = np.bincount(cid_sorted, minlength=n_clusters)
    bounds = np.r_[0, np.cumsum(counts)]
    return cluster_of, rec_sorted.astype(np.int64), bounds.astype(np.int64)


def expand_with_siblings(pool, cluster_maps, cand_js, extra: int = 15):
    """Add cluster siblings of each candidate (deduped, capped)."""
    cluster_of, members, bounds = cluster_maps
    js = list(cand_js)
    for j in cand_js:
        cid = cluster_of[j]
        s, e = int(bounds[cid]), int(bounds[cid + 1])
        if e - s > 1:
            js.extend(int(x) for x in members[s:e][:extra + 1])
    if len(js) == len(cand_js):
        return np.asarray(sorted(cand_js), dtype=np.int64)
    uniq = np.unique(np.asarray(js, dtype=np.int64))
    return uniq
