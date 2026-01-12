# train_ml.py  (FINAL – ULTRA-SMALL DATA SAFE)

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ===============================
# 0. LOAD DATA
# ===============================

X = np.load("../data/language/processed/X_features.npy")
y = np.load("../data/language/processed/y_labels.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Class distribution:", np.bincount(y))

# ===============================
# 1. Define pipeline
# ===============================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear"
    ))
])

# ===============================
# 2. Calibrated model
# ===============================
# NOTE: cv=2 is the MINIMUM possible, and even this is fragile
# We do NOT attempt CV-based evaluation

calibrated_model = CalibratedClassifierCV(
    estimator=pipeline,
    method="sigmoid",
    cv=2
)

# ===============================
# 3. Fit on FULL DATA ONLY
# ===============================

calibrated_model.fit(X, y)

# ===============================
# 4. Save model
# ===============================

joblib.dump(calibrated_model, "../models/language_risk_model.pkl")
print("✅ Calibrated language risk model saved")
