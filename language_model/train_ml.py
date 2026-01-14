import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# LOAD DATA
PROJECT_ROOT = Path(__file__).resolve().parent.parent
X_PATH = PROJECT_ROOT / "data" / "language" / "processed" / "X_features.npy"
Y_PATH = PROJECT_ROOT / "data" / "language" / "processed" / "y_labels.npy"
MODEL_PATH = PROJECT_ROOT / "models" / "language_risk_model.pkl"

X = np.load(X_PATH)
y = np.load(Y_PATH)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Class distribution:", np.bincount(y))


assert X.ndim == 2, "X must be 2D"
assert len(X) == len(y), "X and y length mismatch"

EXPECTED_FEATURES = 8  
assert X.shape[1] == EXPECTED_FEATURES, (
    f"Expected {EXPECTED_FEATURES} features, got {X.shape[1]}"
)

# Warn if dataset is dangerously small
if np.min(np.bincount(y)) < 3:
    print("WARNING: Extremely small minority class. Calibration may be unstable.")

# pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear"
    ))
])


# Calibrated model
calibrated_model = CalibratedClassifierCV(
    estimator=pipeline,
    method="sigmoid",
    cv=2
)

# Fit on FULL DATA ONLY
calibrated_model.fit(X, y)

# 4. Save model
joblib.dump(calibrated_model, MODEL_PATH)
print("Calibrated language risk model saved")
print(MODEL_PATH)
