import logging
from typing import Optional

from language_model.text_features import extract_text_features

logger = logging.getLogger(__name__)

# Feature weights (must sum to 1.0)
_WEIGHTS = {
    "non_word_ratio":       0.40,
    "phonetic_error_ratio": 0.25,
    "rare_word_ratio":      0.10,
    "perplexity":           0.25,
}


def predict_language_risk(text: str) -> Optional[float]:
    """
    Compute a dyslexia language-risk score from OCR text.

    Parameters
    ----------
    text : str
        Raw text extracted from the handwriting image.

    Returns
    -------
    float in [0, 1], or None if text is too short to be meaningful.
    """
    words = text.split()
    if len(words) < 10:
        logger.info("Text too short (%d words) for language risk scoring.", len(words))
        return None

    features = extract_text_features(text)

    risk = sum(
        _WEIGHTS[key] * features[key]
        for key in _WEIGHTS
    )
    risk = max(0.0, min(1.0, risk))

    logger.info(
        "Language risk: %.3f  (nwr=%.2f  per=%.2f  rwr=%.2f  ppl=%.2f)",
        risk,
        features["non_word_ratio"],
        features["phonetic_error_ratio"],
        features["rare_word_ratio"],
        features["perplexity"],
    )
    return risk
