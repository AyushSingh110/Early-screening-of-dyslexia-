"""
Linguistic feature extraction for dyslexia risk scoring.

Extracts features from OCR text that correlate with dyslexic writing patterns:
  - Non-word ratio (spelling errors)
  - Phonetic substitution ratio
  - Rare / low-frequency word ratio
  - Language model perplexity (DistilGPT2, lazy-loaded)
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded heavy dependencies
# ---------------------------------------------------------------------------
_lm_model     = None
_lm_tokenizer = None
_spell_checker = None


def _get_spell_checker():
    global _spell_checker
    if _spell_checker is None:
        try:
            import enchant
            _spell_checker = enchant.Dict("en_US")
            logger.info("PyEnchant spell-checker loaded (en_US)")
        except Exception as exc:
            logger.warning("PyEnchant unavailable (%s) — using basic fallback", exc)
            _spell_checker = False  # sentinel: tried but failed
    return _spell_checker if _spell_checker is not False else None


def _get_lm():
    global _lm_model, _lm_tokenizer
    if _lm_model is None:
        try:
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
            import torch
            _lm_tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
            _lm_model     = GPT2LMHeadModel.from_pretrained("distilgpt2")
            _lm_model.eval()
            logger.info("DistilGPT2 loaded for perplexity scoring")
        except Exception as exc:
            logger.warning("DistilGPT2 unavailable (%s)", exc)
            _lm_model = False
    return (_lm_tokenizer, _lm_model) if _lm_model is not False else (None, None)


# ---------------------------------------------------------------------------
# Common English words for rare-word detection (top ~2 000)
# ---------------------------------------------------------------------------
_COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
    "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
    "about", "who", "get", "which", "go", "me", "when", "make", "can", "like",
    "time", "no", "just", "him", "know", "take", "people", "into", "year",
    "your", "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most",
    "us", "great", "between", "need", "large", "often", "hand", "high",
    "place", "hold", "turn", "here", "why", "help", "put", "different",
    "away", "again", "off", "tell", "boy", "follow", "came", "show", "also",
    "around", "form", "small", "set", "put", "end", "does", "another",
    "well", "large", "must", "big", "even", "such", "because", "turn",
    "here", "why", "asked", "went", "men", "read", "need", "land", "said",
    "each", "she", "which", "do", "been", "call", "who", "am", "its", "now",
    "find", "long", "down", "day", "did", "made", "may", "part", "school",
    "still", "learn", "plant", "cover", "food", "sun", "four", "between",
    "state", "never", "became", "same", "before", "let", "where",
}


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _non_word_ratio(words: List[str]) -> float:
    """Fraction of words that fail spell-check."""
    if not words:
        return 0.0
    checker = _get_spell_checker()
    if checker is None:
        # Fallback: flag words longer than 12 chars or with odd char patterns
        errors = sum(
            1 for w in words
            if len(w) > 12 or re.search(r"(.)\1{2,}", w)
        )
        return errors / len(words)
    errors = sum(1 for w in words if len(w) > 1 and not checker.check(w))
    return errors / len(words)


def _phonetic_error_ratio(words: List[str]) -> float:
    """
    Fraction of words whose phonetic code matches a known word but the
    spelling itself is wrong (classic dyslexic substitution, e.g. 'fone' for 'phone').
    """
    try:
        from metaphone import doublemetaphone
    except ImportError:
        return 0.0

    checker = _get_spell_checker()
    if checker is None:
        return 0.0

    phonetic_errors = 0
    for word in words:
        if len(word) <= 2:
            continue
        if not checker.check(word):
            suggestions = checker.suggest(word)
            if suggestions:
                orig_code = doublemetaphone(word)
                for sug in suggestions[:3]:
                    sug_code = doublemetaphone(sug)
                    if orig_code[0] and orig_code[0] == sug_code[0]:
                        phonetic_errors += 1
                        break
    return phonetic_errors / max(len(words), 1)


def _rare_word_ratio(words: List[str]) -> float:
    """Fraction of words not in the common-word list (proxy for unusual vocabulary)."""
    if not words:
        return 0.0
    rare = sum(1 for w in words if w not in _COMMON_WORDS and len(w) > 3)
    return rare / len(words)


def _perplexity(text: str) -> float:
    """DistilGPT2 perplexity of the text (higher = less fluent)."""
    tokenizer, model = _get_lm()
    if model is None or len(text.split()) < 5:
        return 50.0  # neutral default

    import torch
    try:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        return float(torch.exp(loss).item())
    except Exception as exc:
        logger.warning("Perplexity computation failed: %s", exc)
        return 50.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_features(text: str) -> Dict[str, float]:
    """
    Return a dict of linguistic features relevant to dyslexia screening.

    Keys
    ----
    non_word_ratio        : fraction of misspelled words          [0, 1]
    phonetic_error_ratio  : fraction of phonetic substitutions    [0, 1]
    rare_word_ratio       : fraction of uncommon words            [0, 1]
    perplexity            : LM perplexity (normalised to [0, 1])  [0, 1]
    word_count            : raw word count
    avg_word_length       : mean characters per word
    """
    words = _tokenize(text)

    nwr  = _non_word_ratio(words)
    per  = _phonetic_error_ratio(words)
    rwr  = _rare_word_ratio(words)
    ppl  = _perplexity(text)
    ppl_norm = min(ppl / 500.0, 1.0)  # normalise: ~500 is very high perplexity

    avg_len = sum(len(w) for w in words) / max(len(words), 1)

    features = {
        "non_word_ratio":       nwr,
        "phonetic_error_ratio": per,
        "rare_word_ratio":      rwr,
        "perplexity":           ppl_norm,
        "word_count":           float(len(words)),
        "avg_word_length":      avg_len,
    }

    logger.debug("Text features: %s", features)
    return features
