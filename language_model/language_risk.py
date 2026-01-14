import joblib
import numpy as np
from pathlib import Path

from language_model.feature_engineering import extract_language_features


# Load calibrated model once
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "language_risk_model.pkl"

print("Language risk model loaded successfully")

# Public API
def predict_language_risk(raw_features: dict, debug: bool = True):
    """
    Returns calibrated language-based dyslexia risk (0–1)
    """

    print("\nReceived raw language features:")
    for k, v in raw_features.items():
        print(f"  {k:25s}: {v}")

    X = extract_language_features(raw_features)

    prob = language_model.predict_proba(X)[0, 1]

    
    # DEBUG OUTPUT
    if debug:
        print("\nLanguage Risk Prediction")
        print(f"Calibrated language risk: {prob:.4f}")

    return float(prob)



# Standalone test 
if __name__ == "__main__":
    test_features = {
        "spelling_error_rate": 0.42,
        "non_word_ratio": 0.25,
        "phonetic_error_ratio": 0.30,
        "repetition_score": 0.20,
        "avg_word_length": 4.6,
        "word_count": 128,
        "rare_word_ratio": 0.18
    }

    risk = predict_language_risk(test_features, debug=True)
    print("\nLangauage risk executed")
