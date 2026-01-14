import math
from language_model.text_features import compute_raw_language_features


# Softening helpers
def log_soften(x):
    return math.log1p(max(x, 0.0))

def cap(x, max_val):
    return min(x, max_val)

def soften_perplexity(p):
    p = min(p, 500.0)
    return math.log1p(p) / math.log1p(500.0)

# Main API
def predict_language_risk(text: str, debug=False):
    raw = compute_raw_language_features(text)
    if raw is None:
        return None

    # Softened features
    spelling = log_soften(cap(raw["spelling_error_rate"], 0.5))
    non_word = log_soften(cap(raw["non_word_ratio"], 0.4))
    phonetic = log_soften(cap(raw["phonetic_error_ratio"], 0.4))
    repetition = log_soften(cap(raw["repetition_score"], 0.4))
    rare = log_soften(cap(raw["rare_word_ratio"], 0.4))
    perplexity = soften_perplexity(raw["lm_perplexity"])

    # Severity score
    severity = (
        0.35 * spelling +
        0.15 * non_word +
        0.10 * phonetic +
        0.25 * repetition +
        0.05 * rare +
        0.10 * perplexity
    )

    severity = min(max(severity, 0.0), 1.0)

    if debug:
        print("Language severity:", severity)

    return severity
