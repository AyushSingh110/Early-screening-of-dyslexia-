import numpy as np
import math

# softening functions
def log_soften(x):
    """Log transform to reduce dominance of extreme values"""
    return math.log1p(max(x, 0.0))

def cap(x, max_val):
    """Cap extreme pathological values"""
    return min(x, max_val)

def soften_perplexity(p):
    p = min(p, 500.0)       
    return math.log1p(p)    

# Main feature extraction
def extract_language_features(raw_features: dict):
    """
    raw_features: dict with keys:
        spelling_error_rate
        non_word_ratio
        phonetic_error_ratio
        repetition_score
        avg_word_length
        word_count
        rare_word_ratio,
        lm_perplexity
    """

    #  Cap extreme values (linguistically justified) ---
    spelling_error_rate = cap(raw_features["spelling_error_rate"], 0.5)
    non_word_ratio = cap(raw_features["non_word_ratio"], 0.4)
    phonetic_error_ratio = cap(raw_features["phonetic_error_ratio"], 0.4)
    repetition_score = cap(raw_features["repetition_score"], 0.4)
    rare_word_ratio = cap(raw_features["rare_word_ratio"], 0.4)

    #  Log soften ratios 
    spelling_error_rate = log_soften(spelling_error_rate)
    non_word_ratio = log_soften(non_word_ratio)
    phonetic_error_ratio = log_soften(phonetic_error_ratio)
    repetition_score = log_soften(repetition_score)
    rare_word_ratio = log_soften(rare_word_ratio)
    
    lm_perplexity = soften_perplexity(raw_features["lm_perplexity"])
    #  Normalize word count 
    word_count = raw_features["word_count"] / 100.0
    avg_word_length = raw_features["avg_word_length"]

    #  Final feature vector 
    features = np.array([
        spelling_error_rate,
        non_word_ratio,
        phonetic_error_ratio,
        repetition_score,
        avg_word_length,
        word_count,
        rare_word_ratio,
        lm_perplexity
    ]).reshape(1, -1)
    print(features)
    return features

if __name__ == "__main__":
    # Dummy test input
    raw_features = {
        "spelling_error_rate": 0.42,
        "non_word_ratio": 0.25,
        "phonetic_error_ratio": 0.30,
        "repetition_score": 0.20,
        "avg_word_length": 4.6,
        "word_count": 128,
        "rare_word_ratio": 0.18,
        "lm_perplexity": 120.0
    }

    features = extract_language_features(raw_features)
    print("\nFeature extraction completed")
    print("Extracted features:", features)
    print("Shape:", features.shape)

