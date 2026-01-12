# language_model/inference.py

import joblib
import pickle
import numpy as np
from .feature_extraction import extract_features_from_text
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "language_risk_model.pkl")

BASE_DIR = os.path.dirname(__file__)
WORD_FREQ_PATH = os.path.join(BASE_DIR, "word_freq.pkl")


language_model = joblib.load(MODEL_PATH)

def predict_language_risk(text: str) -> float:
    features = extract_features_from_text(text)
    X = np.array(features).reshape(1, -1)
    prob = language_model.predict_proba(X)[0][1]
    return float(prob)
