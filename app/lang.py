"""Lightweight language identification for the four N-ATLaS languages.

Off-the-shelf detectors are unreliable on Hausa, Igbo and Yoruba, so this uses
two cheap signals per language: distinctive characters and common function words.
Each language gets a score; the highest wins. Short or ambiguous text returns "unknown".
"""
import re
import unicodedata

LANGS = ["en", "ha", "ig", "yo"]
NAMES = {"en": "English", "ha": "Hausa", "ig": "Igbo", "yo": "Yoruba"}

# Characters that are strong evidence for one language.
CHARS = {
    "ha": set("ɓɗƙƴƁƊƘƳ"),
    "ig": set("ịụṅỊỤṄ"),
    "yo": set("ẹṣẸṢ"),
}

# Common function words (lowercased, accents stripped where the language is often typed without them).
WORDS = {
    "en": {"the", "is", "and", "of", "to", "in", "you", "what", "how", "are", "this", "that", "it", "for", "with", "can", "my", "i"},
    "ha": {"da", "na", "ne", "ce", "ba", "wannan", "yana", "kuma", "cikin", "shi", "ita", "menene", "yaya", "ina", "zan", "kai", "akwai", "su", "mu", "domin"},
    "ig": {"nke", "bu", "ya", "ka", "ndi", "maka", "gi", "anyi", "nwere", "ihe", "kedu", "gini", "ebee", "mana", "nna", "nne", "dị", "bụ", "ndị", "gị", "ọ"},
    "yo": {"ni", "ti", "ati", "won", "mo", "ko", "si", "naa", "kini", "bawo", "se", "fun", "pelu", "awon", "eyi", "ohun", "je", "wa", "ẹ", "ṣe", "àti", "wọn", "kò", "sí", "náà"},
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def detect(text: str) -> str:
    """Return one of en/ha/ig/yo, or 'unknown' if the evidence is too weak."""
    if not text or not text.strip():
        return "unknown"
    lowered = text.lower()
    tokens = re.findall(r"[^\W\d_]+", lowered)
    plain_tokens = {_strip_accents(t) for t in tokens}
    raw_tokens = set(tokens)

    scores = {lang: 0.0 for lang in LANGS}
    for lang, chars in CHARS.items():
        scores[lang] += 3 * sum(1 for c in text if c in chars)
    for lang, words in WORDS.items():
        plain_words = {_strip_accents(w) for w in words}
        scores[lang] += len(raw_tokens & words) + len(plain_tokens & plain_words)

    # Yoruba tone marks on vowels (grave/acute) without Igbo dot-below letters lean Yoruba.
    if re.search(r"[àáèéìíòóùú]", lowered) and scores["ig"] < scores["yo"] + 1:
        scores["yo"] += 1

    best = max(scores, key=scores.get)
    if scores[best] < 2:
        return "unknown"
    return best
