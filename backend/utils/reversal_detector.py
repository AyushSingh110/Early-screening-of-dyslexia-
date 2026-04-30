import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# Reversal character pairs
REVERSAL_PAIRS: List[Tuple[str, str]] = [
    ('b', 'd'), ('d', 'b'),
    ('p', 'q'), ('q', 'p'),
    ('n', 'u'), ('u', 'n'),
    ('m', 'w'), ('w', 'm'),
]

# Curated dictionary: known dyslexic reversal errors → correct word
KNOWN_REVERSAL_WORDS: Dict[str, str] = {
    # b ↔ d
    'doy':  'boy',  'dook': 'book', 'dox':  'box',  'ded':  'bed',
    'dat':  'bat',  'dite': 'bite', 'dug':  'bug',  'durn': 'burn',
    'dill': 'bill', 'dend': 'bend', 'dand': 'band', 'dack': 'back',
    'bog':  'dog',  'boor': 'door', 'bown': 'down', 'brag': 'drag',
    'braw': 'draw', 'brop': 'drop', 'bry':  'dry',  'bisk': 'disk',
    'birt': 'dirt', 'boze': 'doze', 'bive': 'dive', 'bust': 'dust',
    # p ↔ q
    'qin':  'pin',  'qot':  'pot',  'qan':  'pan',  'qig':  'pig',
    'puit': 'quit', 'puick':'quick','pueen':'queen',
    # n ↔ u
    'ub':   'up',   'uder': 'under','ud':   'up',   'uext': 'next',
    # m ↔ w
    'wom':  'mom',  'wade': 'made', 'wan':  'man',  'wap':  'map',
    'wuch': 'much', 'wake': 'make', 'wail': 'mail', 'woon': 'moon',
    # Common whole-word reversals
    'saw':  'was',  'on':   'no',   'pot':  'top',  'god':  'dog',
    'tip':  'pit',  'nap':  'pan',  'net':  'ten',  'gum':  'mug',
    'bud':  'dub',  'sub':  'bus',
}



# Helpers
def _substitute(word: str, from_char: str, to_char: str) -> str:
    return word.replace(from_char, to_char)


def _is_real_word(word: str, checker) -> bool:
    if checker is None:
        return False
    try:
        return checker.check(word)
    except Exception:
        return False



# Public API
def detect_reversals(text: str, spell_checker=None) -> Dict:
    """
    Detect letter reversals in OCR-extracted text.

    Parameters
    ----------
    text          : raw OCR string from the handwriting image
    spell_checker : optional PyEnchant Dict object for spell-checking

    Returns
    -------
    dict
      reversal_count  : number of words with detected reversals
      reversal_ratio  : reversal_count / total_words  ∈ [0, 1]
      reversal_words  : list of (misspelled, likely_intended) tuples (max 10)
      total_words     : total word count in text
    """
    words = re.findall(r'[a-zA-Z]+', text.lower())
    if not words:
        return {
            'reversal_count': 0,
            'reversal_ratio': 0.0,
            'reversal_words': [],
            'total_words':    0,
        }

    reversal_words: List[Tuple[str, str]] = []
    seen: set = set()

    for word in words:
        if len(word) < 2 or word in seen:
            continue
        seen.add(word)

        # 1. Check curated dictionary first (fast + high confidence)
        if word in KNOWN_REVERSAL_WORDS:
            reversal_words.append((word, KNOWN_REVERSAL_WORDS[word]))
            continue

        # 2. Spell-checker based substitution detection
        if spell_checker is not None:
            if not _is_real_word(word, spell_checker):
                for from_c, to_c in REVERSAL_PAIRS:
                    if from_c in word:
                        candidate = _substitute(word, from_c, to_c)
                        if candidate != word and _is_real_word(candidate, spell_checker):
                            reversal_words.append((word, candidate))
                            break

    total    = len(words)
    count    = len(reversal_words)
    ratio    = count / max(total, 1)

    logger.debug("Reversal detection: %d reversals in %d words (%.1f%%)", count, total, ratio * 100)

    return {
        'reversal_count': count,
        'reversal_ratio': round(ratio, 4),
        'reversal_words': reversal_words[:10],
        'total_words':    total,
    }


def reversal_risk_score(reversal_info: Dict) -> float:
    """
    Convert reversal detection output to a [0, 1] risk score.

    Thresholds based on literature:
      < 2%  → negligible
      2-8%  → mild indicator
      > 8%  → strong indicator
    """
    ratio = reversal_info.get('reversal_ratio', 0.0)
    # Scale: 0% → 0.0,  8%+ → 1.0
    return min(ratio / 0.08, 1.0)
