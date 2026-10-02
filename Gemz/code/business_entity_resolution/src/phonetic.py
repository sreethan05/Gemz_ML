"""Indic phonetic collapse module for business entity matching.

Normalizes Indic-transliteration noise:
- ph -> f, kh -> k, c -> k, q -> k, v -> w, z -> j
- Strip 'h' after consonants (bh->b, dh->d, th->t, gh->g, sh->s, etc.)
- Drop vowels after the first character of each token
- Collapse doubled letters
"""
import re

_CONSONANTS = "bcdfghjklmnpqrstvwxyz"
_VOWELS = set("aeiou")
_CONSONANT_H = re.compile(r"([bcdfghjklmnpqrstvwxyz])h+", re.IGNORECASE)
_DOUBLES = re.compile(r"([a-z])\1+", re.IGNORECASE)


def phonetic_collapse(s: str) -> str:
    """Normalize transliteration noise in an entity string."""
    if not s:
        return ""
    words = s.lower().split()
    out_words = []
    for w in words:
        # Strip non-alphanumeric chars
        w = re.sub(r"[^a-z0-9]", "", w)
        if not w:
            continue
        if w.isdigit():
            out_words.append(w)
            continue
        
        # 1. Phonetic substitutions
        w = w.replace("ph", "f")
        w = w.replace("kh", "k")
        w = w.replace("c", "k")
        w = w.replace("q", "k")
        w = w.replace("v", "w")
        w = w.replace("z", "j")
        
        # 2. Strip 'h' after consonants (e.g., bh->b, dh->d, th->t, sh->s, gh->g)
        w = _CONSONANT_H.sub(r"\1", w)
        
        # 3. Collapse doubled letters
        w = _DOUBLES.sub(r"\1", w)
        
        # 4. Drop vowels after the first character
        if len(w) > 1:
            w = w[0] + "".join(c for c in w[1:] if c not in _VOWELS)
            
        # 5. Collapse doubled letters again (if vowel drop created adjacency)
        w = _DOUBLES.sub(r"\1", w)
        
        if w:
            out_words.append(w)
            
    return " ".join(out_words)
