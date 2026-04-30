import os
from pathlib import Path

# System Information
VERSION = "1.0.0"
SYSTEM_NAME = "Dyslexia Handwriting Screening System"

# Model Configuration
# Image input size (ResNet50 default is 224x224)
IMAGE_SIZE = 224

# Model architecture
MODEL_ARCHITECTURE = "resnet50"
NUM_CLASSES = 1  

# Project paths
BACKEND_DIR = Path(__file__).resolve().parent
BASE_DIR = BACKEND_DIR.parent
MODELS_DIR = Path(os.getenv("DYSLEXIA_MODELS_DIR", BASE_DIR / "models"))
DATA_DIR = Path(os.getenv("DYSLEXIA_DATA_DIR", BASE_DIR / "data"))

# Class indices (ImageFolder sorts alphabetically: dyslexic < normal)
DYSLEXIC_CLASS_IDX = 0   # label=0 in all datasets and CSVs
NORMAL_CLASS_IDX   = 1   # label=1
# Model output is P(normal). P(dyslexic) = 1 - model_output

# Trained model checkpoint paths (absolute so they resolve from any working dir)
BASE_MODEL_PATH = str(MODELS_DIR / "resnet50_dyslexia_base.pth")
FINETUNED_MODEL_PATH = str(MODELS_DIR / "resnet50_dyslexia_finetuned.pth")
INFERENCE_MODEL_PATH = FINETUNED_MODEL_PATH


# Training Configuration
# Dataset paths
TRAIN_DIR = str(DATA_DIR / "train")
VAL_DIR = str(DATA_DIR / "val")
TEST_DIR = str(DATA_DIR / "test")

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
WEIGHT_DECAY = 0.01

# Data augmentation
RANDOM_ROTATION_DEGREES = 10
HORIZONTAL_FLIP_PROB = 0.5
COLOR_JITTER_ENABLED = False


# Inference Configuration

# Prediction thresholds
DYSLEXIA_THRESHOLD = 0.5  # Probability threshold for classification
HIGH_RISK_THRESHOLD = 0.5  # Risk score threshold for high risk
MODERATE_RISK_THRESHOLD = 0.3  # Risk score threshold for moderate risk

# Patch extraction
PATCH_SIZE = IMAGE_SIZE
PATCH_STRIDE = IMAGE_SIZE 
PATCH_MIN_VARIANCE = 100.0  # Minimum variance to consider patch valid

# Performance
USE_MIXED_PRECISION = True
NUM_WORKERS = 0 

# Logging Configuration
LOG_DIR = Path(os.getenv("DYSLEXIA_LOG_DIR", BASE_DIR / "logs"))
LOG_LEVEL = "INFO"  

# Enable/disable logging to file
LOG_TO_FILE = True
LOG_FILE = LOG_DIR / "dyslexia_system.log"
HISTORY_DB_PATH = Path(os.getenv("DYSLEXIA_HISTORY_DB", BASE_DIR / "screening_history.db"))


# UI Configuration

# Streamlit settings
MAX_UPLOAD_SIZE_MB = 10
SUPPORTED_FORMATS = ["png", "jpg", "jpeg"]

# Results display
SHOW_DETAILED_STATS = True
SHOW_CONFIDENCE_DISTRIBUTION = True
MIN_PATCHES_FOR_RELIABLE_SCREENING = 5

# Device Configuration
# Auto-detect CUDA availability
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU = torch.cuda.is_available()


# Model Performance Metrics
# Expected performance benchmarks
BASELINE_ACCURACY = 0.75
BASELINE_RECALL = 0.76
BASELINE_PRECISION = 0.74

# Ethical Guidelines
DISCLAIMER_TEXT = (
    "This tool provides early screening support only and is NOT a medical diagnosis. "
    "It uses AI to identify handwriting patterns that may be associated with dyslexia. "
    "Always consult qualified educational psychologists or medical professionals for "
    "comprehensive assessment and diagnosis."
)

# Create necessary directories
def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [MODELS_DIR, DATA_DIR, LOG_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Auto-setup on import
setup_directories()
