"""v5_optimized features: Rich pair featurization for high-precision matching"""
from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz import process as rfp
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import Config, Record


class PairFeaturizer:
    """Enhanced pair featurization with 60+ features"""
    
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
    
    def fit(self, records: list[Record]) -> "PairFeaturizer":
        self.name_vec.fit([r.name_norm or " " for r in records])
        self.addr_vec.fit([r.addr_norm or " " for r in records])
        return self
    
    @property
    def feature_names(self) -> list[str]:
        return [
            # Name similarities (15)
            "name_jw", "name_lev", "name_sort", "name_set", "name_partial",
            "name_tfidf", "name_jaccard", "name_contain", "name_lenratio",
            "name_prefix_eq", "name_suffix_eq", "name_first_eq", "name_last_eq",
            "name_exact", "name_core_eq",
            
            # Address similarities (15)
            "addr_jw", "addr_lev", "addr_sort", "addr_set", "addr_partial",
            "addr_tfidf", "addr_jaccard", "addr_lenratio",
            "addr_prefix_eq", "addr_suffix_eq", "addr_num_overlap",
            "addr_pin_eq", "addr_pin_both", "addr_exact", "addr_core_eq",
            
            # Cross features (10)
            "name_sort_x_addr_sort", "name_set_x_addr_set",
            "name_jw_x_addr_jw", "name_sort_x_num_ov",
            "addr_set_x_name_jw", "name_contain_x_addr_contain",
            "country_eq", "src_is_s2",
            
            # Structural (10)
            "num_overlap", "num_a_count", "num_b_count",
            "pin_match", "pin_conflict", "pin_both", "num_conflict",
            "first_word_match", "name_len_diff", "addr_len_diff",
            
            # Interaction features (10)
            "prod_sort", "min_set", "mean_set",
            "name_sort_x_num", "addr_set_x_name_jw",
            "name_jw_x_addr_sort", "name_set_x_addr_jw",
            "num_ov_x_name_sort", "pin_match_x_name_jw",
            
            # New: Character n-gram Jaccard (4-gram)
            "name_char4_jaccard", "addr_char4_jaccard",
            "name_char3_jaccard", "addr_char3_jaccard",
        ]
    
    def _char_ngram_jaccard(self, a: str, b: str, n: int) -> float:
        if not a or not b or len(a) < n or len(b) < n:
            return 0.0
        ga = {a[i:i+n] for i in range(len(a) - n + 1)}
        gb = {b[i:i+n] for i in range(len(b) - n + 1)}
        u = ga | gb
        return len(ga & gb) / len(u) if u else 0.0
    
    def transform(self, records: list[Record],
                  pairs: list[tuple[int, int]],
                  log_every: int = 200_000) -> np.ndarray:
        n = len(pairs)
        if n == 0:
            return np.zeros((0, len(self.feature_names)), dtype=np.float32)
        
        # Pre-compute TF-IDF cosine
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
        
        for k, (i, j) in enumerate(pairs):
            a, b = records[i], records[j]
            has_name = bool(a.name_norm) and bool(b.name_norm)
            has_addr = bool(a.addr_norm) and bool(b.addr_norm)
            an, bn = a.name_norm or " ", b.name_norm or " "
            aa, ba = a.addr_norm or " ", b.addr_norm or " "
            
            # --- Name similarities ---
            if has_name:
                n_jw = jw(an, bn)
                n_lev = lv(an, bn)
                n_sort = tsr(an, bn) / 100.0
                n_set = tset(an, bn) / 100.0
                n_part = part(an, bn) / 100.0
                n_tfidf = name_cos[k]
                
                core_u = len(a.core_tokens | b.core_tokens)
                n_jac = len(a.core_tokens & b.core_tokens) / core_u if core_u else 0.0
                n_cont = (len(a.core_tokens & b.core_tokens) / 
                         min(len(a.core_tokens), len(b.core_tokens))) if (a.core_tokens and b.core_tokens) else 0.0
                n_len = min(len(an), len(bn)) / max(len(an), len(bn), 1)
                
                n_pref = 1.0 if (an[:3] == bn[:3] and len(an) >= 3 and len(bn) >= 3) else 0.0
                n_suff = 1.0 if (an[-3:] == bn[-3:] and len(an) >= 3 and len(bn) >= 3) else 0.0
                n_first = 1.0 if (an.split()[0] == bn.split()[0] if an.split() and bn.split() else False) else 0.0
                n_last = 1.0 if (an.split()[-1] == bn.split()[-1] if an.split() and bn.split() else False) else 0.0
                n_exact = 1.0 if an == bn else 0.0
                n_core_eq = 1.0 if (a.core_tokens and a.core_tokens == b.core_tokens) else 0.0
            else:
                n_jw = n_lev = n_sort = n_set = n_part = n_tfidf = 0.0
                n_jac = n_cont = n_len = 0.0
                n_pref = n_suff = n_first = n_last = n_exact = n_core_eq = 0.0
            
            # --- Address similarities ---
            if has_addr:
                a_jw = jw(aa, ba)
                a_lev = lv(aa, ba)
                a_sort = tsr(aa, ba) / 100.0
                a_set = tset(aa, ba) / 100.0
                a_part = part(aa, ba) / 100.0
                a_tfidf = addr_cos[k]
                
                addr_u = len(a.addr_tokens | b.addr_tokens)
                a_jac = len(a.addr_tokens & b.addr_tokens) / addr_u if addr_u else 0.0
                a_len = min(len(aa), len(ba)) / max(len(aa), len(ba), 1)
                
                a_pref = 1.0 if (aa[:3] == ba[:3] and len(aa) >= 3 and len(ba) >= 3) else 0.0
                a_suff = 1.0 if (aa[-3:] == ba[-3:] and len(aa) >= 3 and len(ba) >= 3) else 0.0
                
                num_u = len(set(a.numbers) | set(b.numbers))
                num_ov = len(set(a.numbers) & set(b.numbers)) / num_u if num_u else 0.0
                
                pin_match = 1.0 if (a.pins and b.pins and (a.pins & b.pins)) else 0.0
                pin_conflict = 1.0 if (a.pins and b.pins and not (a.pins & b.pins)) else 0.0
                pin_both = 1.0 if (a.pins and b.pins) else 0.0
                
                a_exact = 1.0 if aa == ba else 0.0
                a_core_eq = 1.0 if (a.addr_tokens and a.addr_tokens == b.addr_tokens) else 0.0
            else:
                a_jw = a_lev = a_sort = a_set = a_part = a_tfidf = 0.0
                a_jac = a_len = 0.0
                a_pref = a_suff = 0.0
                num_ov = 0.0
                pin_match = pin_conflict = pin_both = 0.0
                a_exact = a_core_eq = 0.0
            
            # --- Cross features ---
            country_eq = 1.0 if a.country == b.country else 0.0
            src_is_s2 = 1.0 if b.entity_id.startswith("S2-") else 0.0
            
            # --- Structural ---
            a_first = an.split(" ", 1)[0] if an.strip() else ""
            b_first = bn.split(" ", 1)[0] if bn.strip() else ""
            first_word_match = 1.0 if (a_first and a_first == b_first) else 0.0
            name_len_diff = abs(len(an) - len(bn)) / max(len(an), len(bn), 1)
            addr_len_diff = abs(len(aa) - len(ba)) / max(len(aa), len(ba), 1)
            
            # --- Interaction features ---
            prod_sort = n_sort * a_sort
            min_set = min(n_set, a_set)
            mean_set = (n_set + a_set) / 2.0
            name_sort_x_num = n_sort * num_ov if num_ov else 0.0
            addr_set_x_name_jw = a_set * n_jw
            name_jw_x_addr_sort = n_jw * a_sort
            name_set_x_addr_jw = n_set * a_jw
            num_ov_x_name_sort = num_ov * n_sort
            pin_match_x_name_jw = pin_match * n_jw
            
            # --- Char n-gram Jaccard ---
            name_c4 = self._char_ngram_jaccard(an.replace(" ", ""), bn.replace(" ", ""), 4)
            addr_c4 = self._char_ngram_jaccard(aa.replace(" ", ""), ba.replace(" ", ""), 4)
            name_c3 = self._char_ngram_jaccard(an.replace(" ", ""), bn.replace(" ", ""), 3)
            addr_c3 = self._char_ngram_jaccard(aa.replace(" ", ""), ba.replace(" ", ""), 3)
            
            # --- Fill row ---
            row = X[k]
            idx = 0
            row[idx] = n_jw; idx += 1
            row[idx] = n_lev; idx += 1
            row[idx] = n_sort; idx += 1
            row[idx] = n_set; idx += 1
            row[idx] = n_part; idx += 1
            row[idx] = n_tfidf; idx += 1
            row[idx] = n_jac; idx += 1
            row[idx] = n_cont; idx += 1
            row[idx] = n_len; idx += 1
            row[idx] = n_pref; idx += 1
            row[idx] = n_suff; idx += 1
            row[idx] = n_first; idx += 1
            row[idx] = n_last; idx += 1
            row[idx] = n_exact; idx += 1
            row[idx] = n_core_eq; idx += 1
            
            row[idx] = a_jw; idx += 1
            row[idx] = a_lev; idx += 1
            row[idx] = a_sort; idx += 1
            row[idx] = a_set; idx += 1
            row[idx] = a_part; idx += 1
            row[idx] = a_tfidf; idx += 1
            row[idx] = a_jac; idx += 1
            row[idx] = a_len; idx += 1
            row[idx] = a_pref; idx += 1
            row[idx] = a_suff; idx += 1
            row[idx] = num_ov; idx += 1
            row[idx] = pin_match; idx += 1
            row[idx] = pin_both; idx += 1
            row[idx] = a_exact; idx += 1
            row[idx] = a_core_eq; idx += 1
            
            row[idx] = prod_sort; idx += 1
            row[idx] = min_set; idx += 1
            row[idx] = mean_set; idx += 1
            row[idx] = name_jw_x_addr_sort; idx += 1
            row[idx] = name_set_x_addr_set; idx += 1
            row[idx] = name_jw_x_addr_jw; idx += 1
            row[idx] = name_sort_x_num_ov; idx += 1
            row[idx] = addr_set_x_name_jw; idx += 1
            row[idx] = name_contain_x_addr_contain; idx += 1
            row[idx] = country_eq; idx += 1
            row[idx] = src_is_s2; idx += 1
            
            row[idx] = num_ov; idx += 1
            row[idx] = len(a.numbers); idx += 1
            row[idx] = len(b.numbers); idx += 1
            row[idx] = pin_match; idx += 1
            row[idx] = pin_conflict; idx += 1
            row[idx] = pin_both; idx += 1
            row[idx] = 1.0 if (a.numbers and b.numbers and not (set(a.numbers) & set(b.numbers))) else 0.0; idx += 1
            row[idx] = first_word_match; idx += 1
            row[idx] = name_len_diff; idx += 1
            row[idx] = addr_len_diff; idx += 1
            
            row[idx] = prod_sort; idx += 1
            row[idx] = min_set; idx += 1
            row[idx] = mean_set; idx += 1
            row[idx] = name_sort_x_num; idx += 1
            row[idx] = addr_set_x_name_jw; idx += 1
            row[idx] = name_jw_x_addr_sort; idx += 1
            row[idx] = name_set_x_addr_jw; idx += 1
            row[idx] = num_ov_x_name_sort; idx += 1
            row[idx] = pin_match_x_name_jw; idx += 1
            
            row[idx] = name_c4; idx += 1
            row[idx] = addr_c4; idx += 1
            row[idx] = name_c3; idx += 1
            row[idx] = addr_c3; idx += 1
            
            if log_every and (k + 1) % log_every == 0:
                print(f"    featurised {k + 1:,}/{n:,} pairs")
        
        return X