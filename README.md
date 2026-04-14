# Early Screening of Dyslexia

An AI-powered multimodal system for early-stage dyslexia screening, combining deep learning-based handwriting analysis with linguistic feature engineering. Built for educators, parents, and clinical support professionals.

> **Disclaimer**: This tool is intended for early screening purposes only and does not constitute a medical diagnosis. Results should be interpreted by qualified professionals.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Explainability](#explainability)
- [Configuration](#configuration)

---

## Overview

Dyslexia is one of the most common learning disabilities, affecting reading, writing, and spelling abilities. Early identification is critical for timely intervention. This system automates the screening process by analyzing two complementary signals:

1. **Handwriting patterns** — using a fine-tuned ResNet-50 convolutional neural network to classify handwriting patches as dyslexic or typical.
2. **Linguistic patterns** — using engineered linguistic features and GPT-2 perplexity to detect writing errors characteristic of dyslexia.

Both modalities are fused into a unified risk score through a weighted multimodal fusion strategy.

---

## System Architecture

```
                    ┌─────────────────────────────────┐
                    │        Uploaded Image            │
                    └────────────┬────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
   ┌──────────▼──────────┐              ┌───────────▼──────────┐
   │   Patch Extraction   │              │    OCR (Tesseract)   │
   │  (224×224 patches)   │              │    Text Extraction   │
   └──────────┬──────────┘              └───────────┬──────────┘
              │                                     │
   ┌──────────▼──────────┐              ┌───────────▼──────────┐
   │  ResNet-50 (Vision)  │              │  Linguistic Feature  │
   │  Binary Classifier   │              │     Extraction       │
   └──────────┬──────────┘              └───────────┬──────────┘
              │                                     │
   ┌──────────▼──────────┐              ┌───────────▼──────────┐
   │  Handwriting Risk    │              │  Language Risk Score  │
   │       Score          │              │  (Rule-based / LR)   │
   └──────────┬──────────┘              └───────────┬──────────┘
              │                                     │
              └──────────────┬──────────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │  Multimodal Fusion   │
                  │  (Weighted Average)  │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │   Risk Assessment    │
                  │  LOW / MODERATE /   │
                  │       HIGH          │
                  └─────────────────────┘
```

---

## Features

- **Patch-based inference** — Full handwriting pages are split into 224×224 patches and analyzed individually, enabling analysis of variable-size inputs.
- **Transfer learning** — ResNet-50 pretrained on ImageNet, fine-tuned on a large handwriting dataset (151,649 images).
- **Linguistic analysis** — Detects spelling errors, non-words, phonetic confusions, repetitions, and rare words using handcrafted NLP features.
- **LLM-based perplexity** — Uses DistilGPT2 to measure text perplexity as a proxy for linguistic fluency.
- **Grad-CAM explainability** — Visualizes model attention on handwriting patches so results are interpretable.
- **Multimodal fusion** — Combines handwriting and language risks with learned weights for a more robust prediction.
- **Calibrated confidence** — Probability calibration (sigmoid method) ensures reliable confidence scores.
- **Interactive web app** — Streamlit-based UI requiring no technical expertise to operate.

---

## Tech Stack

### Deep Learning & Machine Learning
| Library | Version | Purpose |
|---|---|---|
| PyTorch | 2.7.1+cu118 | Deep learning framework |
| TorchVision | 0.22.1+cu118 | ResNet-50, image transforms |
| Scikit-learn | — | Logistic Regression, StandardScaler, calibration, metrics |
| HuggingFace Transformers | — | DistilGPT2 for text perplexity |
| HuggingFace Datasets | — | Wikipedia corpus streaming |

### Computer Vision & Image Processing
| Library | Purpose |
|---|---|
| OpenCV (`cv2`) | Image manipulation, Grad-CAM heatmap overlay |
| PIL / Pillow | Image loading and preprocessing |
| pytesseract | Tesseract OCR wrapper for text extraction from images |

### Natural Language Processing
| Library | Purpose |
|---|---|
| PyEnchant (`enchant`) | English spell-checking dictionary (en_US) |
| Metaphone (`metaphone`) | Double Metaphone phonetic similarity matching |

### Data & Utilities
| Library | Purpose |
|---|---|
| NumPy | Numerical computing and array operations |
| Pandas | CSV dataset loading and manipulation |
| Joblib | Model serialization (`.pkl`) |
| Matplotlib | Visualization |
| Seaborn | Statistical plots |

### Web Application
| Library | Purpose |
|---|---|
| Streamlit | Interactive web UI |

---

## Project Structure

```
Early-screening-of-dyslexia-/
│
├── app.py                          # Streamlit web application (main entry point)
├── config.py                       # Centralized configuration (hyperparameters, paths, thresholds)
├── requirements.txt                # Core Python dependencies
├── student_writing.csv             # Linguistic dataset (token-level error annotations)
│
├── utils/                          # Vision model utilities
│   ├── predict.py                  # ResNet-50 model loading and inference
│   ├── preprocess.py               # Image preprocessing and augmentation pipelines
│   ├── patchify.py                 # Sliding window patch extraction
│   ├── ocr.py                      # Tesseract OCR text extraction
│   └── gradcam.py                  # Grad-CAM explainability visualizations
│
├── language_model/                 # Linguistic analysis module
│   ├── __init__.py
│   ├── feature_extraction.py       # Handcrafted linguistic features (7-dimensional)
│   ├── text_features.py            # LLM-based features (DistilGPT2 perplexity)
│   ├── feature_engineering.py      # Feature normalization, log-transform, softening
│   ├── train_ml.py                 # Logistic Regression classifier training
│   ├── inference.py                # Rule-based language risk scoring
│   ├── language_risk.py            # Calibrated model-based language risk
│   ├── evaluate_ml.py              # Language model evaluation
│   ├── build_language_stats.py     # Wikipedia word frequency dictionary builder
│   ├── build_language_dataset.ipynb
│   └── pdf_to_text.ipynb           # PDF to text conversion
│
├── models/                         # Saved model weights (not included in repo)
│   ├── resnet50_dyslexia_base.pth
│   ├── resnet50_dyslexia_finetuned.pth
│   └── language_risk_model.pkl
│
├── data/                           # Dataset directory (not included in repo)
│   ├── train/dyslexic/ & normal/
│   ├── val/
│   ├── test/dyslexic/ & normal/
│   └── language/
│       ├── raw/dyslexic_like/ & normal/
│       └── processed/
│           ├── X_features.npy
│           └── y_labels.npy
│
├── train.ipynb                     # Vision model training notebook
├── train_ml.ipynb                  # Language model training notebook
├── evaluate.ipynb                  # Baseline model evaluation
├── evaluate_finetuned.ipynb        # Fine-tuned model evaluation
├── evaluate_multimodal.ipynb       # Multimodal evaluation
├── data_fomat.ipynb                # Dataset exploration
└── gradcam.ipynb                   # Grad-CAM exploration
```

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended: NVIDIA RTX 3050 4GB or higher)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on the system

### Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd Early-screening-of-dyslexia-

# 2. Create and activate a virtual environment
conda create -n dyslexia_env python=3.10
conda activate dyslexia_env

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install additional dependencies
pip install transformers datasets pytesseract pyenchant metaphone joblib pdfplumber
```

### Tesseract OCR Installation

| Platform | Command |
|---|---|
| Windows | Download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) |
| Linux | `sudo apt-get install tesseract-ocr` |
| macOS | `brew install tesseract` |

---

## Usage

### Running the Web Application

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser.

**Workflow:**
1. Upload a handwriting image (PNG, JPG, JPEG — max 10MB)
2. Toggle Grad-CAM visualization from the sidebar (optional)
3. Click **Analyze Handwriting**
4. View:
   - Overall risk level (LOW / MODERATE / HIGH)
   - Per-patch predictions and confidence distribution
   - Grad-CAM heatmaps highlighting attention regions
   - OCR-extracted text and linguistic risk score
   - Multimodal fusion breakdown

### Command-line Scripts

**Build word frequency dictionary** (required before language feature extraction):
```bash
python language_model/build_language_stats.py
```

**Train the language risk classifier:**
```bash
python language_model/train_ml.py
```

---

## Models

### Vision Model — ResNet-50

**Architecture:**
```
Input: (B, 3, 224, 224)
  ↓
ResNet-50 Backbone (ImageNet pretrained)
  ↓
Adaptive Average Pooling
  ↓
FC(2048 → 256) + ReLU + Dropout(0.5)
  ↓
FC(256 → 1) + Sigmoid
  ↓
Output: Dyslexia probability ∈ [0, 1]
```

**Transfer Learning Strategy:**

| Stage | Epochs | LR | Unfrozen Layers |
|---|---|---|---|
| Baseline | 10 | 1e-3 | Classifier head only |
| Fine-tuning | 5 | 1e-5 | ResNet layer4 + classifier head |

- **Loss function**: Binary Cross-Entropy (BCELoss)
- **Optimizer**: Adam with weight decay 0.01
- **ImageNet normalization**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Language Model — Logistic Regression

**Pipeline:**
```
Raw Text
  ↓
Tokenization (regex: [a-z]+)
  ↓
PyEnchant spell-check + Metaphone phonetic matching
  ↓
7 handcrafted features + 1 DistilGPT2 perplexity feature
  ↓
Feature softening (log1p transform, capping)
  ↓
StandardScaler normalization
  ↓
Logistic Regression (liblinear, balanced class weights)
  ↓
CalibratedClassifierCV (sigmoid, cv=2)
  ↓
Output: Dyslexia language risk probability ∈ [0, 1]
```

### Multimodal Fusion

```
final_risk = 0.60 × handwriting_risk
           + 0.25 × language_risk
           + 0.15 × max(language_risk − handwriting_risk, 0)
```

- If OCR extracts fewer than 20 words: language component is skipped.
- Safety cap: if `handwriting_risk < 0.2`, cap `final_risk` at 0.45 (conservative).

**Risk Thresholds:**

| Level | Score Range |
|---|---|
| LOW | < 0.30 |
| MODERATE | 0.30 – 0.50 |
| HIGH | > 0.50 |

---

## Dataset

### Vision Dataset

- **Format**: `torchvision.datasets.ImageFolder` structure
- **Classes**: `dyslexic` / `normal`
- **Training size**: 151,649 images
- **Test size**: 56,693 images
- **Patch size**: 224×224 (extracted with sliding window, stride=224)
- **File types**: PNG, JPG, JPEG

### Linguistic Dataset (`student_writing.csv`)

Token-level error annotations from student writing samples:

| Column | Description |
|---|---|
| `text_id` | Student ID (S1, S2, ...) |
| `token` | Written word |
| `correct_token` | Gold-standard correct word |
| `position` | Token position in sentence |
| `target` | 1 = error, 0 = correct |

### Language Corpus

- **Normal samples**: Typical writing text files
- **Dyslexic-like samples**: Text files with dyslexia-characteristic patterns
- **Processed features**: `X_features.npy` (154×8), `y_labels.npy` (154,)
- **Word frequency baseline**: Built from 20,000 Wikipedia documents (streaming via HuggingFace Datasets)

---

## Training

### Vision Model

Run cells in [train.ipynb](train.ipynb):

**Data Augmentation (training only):**
- Random rotation: ±10°
- Random horizontal flip: 50%
- Color jitter: brightness/contrast ±0.2
- Resize to 224×224
- ImageNet normalization

**Saved checkpoints:**
- `models/resnet50_dyslexia_base.pth` — After baseline training
- `models/resnet50_dyslexia_finetuned.pth` — After fine-tuning

### Language Model

1. Extract features from raw text corpus:
   ```bash
   python language_model/feature_extraction.py
   ```

2. Train and save the classifier:
   ```bash
   python language_model/train_ml.py
   ```

Run cells in [train_ml.ipynb](train_ml.ipynb) for interactive training with visualization.

---

## Evaluation

### Vision Model — Baseline Results

Evaluated on 56,693 test images:

| Metric | Dyslexic Class | Normal Class | Overall |
|---|---|---|---|
| Accuracy | — | — | **75.44%** |
| Precision | 0.85 | 0.62 | — |
| Recall | 0.76 | 0.75 | — |
| F1-Score | 0.80 | 0.68 | — |

**Confusion Matrix:**
```
                 Predicted Normal    Predicted Dyslexic
Actual Normal        28,135               9,001
Actual Dyslexic       4,920              14,637
```

**Design priority**: High recall (76%) for dyslexic class to minimize missed cases (false negatives), which is critical in a screening context.

### Evaluation Notebooks

| Notebook | Description |
|---|---|
| [evaluate.ipynb](evaluate.ipynb) | Baseline ResNet-50 evaluation |
| [evaluate_finetuned.ipynb](evaluate_finetuned.ipynb) | Fine-tuned model evaluation |
| [evaluate_multimodal.ipynb](evaluate_multimodal.ipynb) | Vision + language fusion evaluation |

---

## Explainability

### Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM produces visual explanations by:
1. Registering forward and backward hooks on ResNet-50's `layer4`
2. Computing gradients of the predicted class score w.r.t. feature maps
3. Weighting feature maps by global-average-pooled gradients
4. Applying ReLU and upsampling to image resolution
5. Overlaying a JET colormap heatmap on the original patch

**Colormap interpretation:**
- **Red/Yellow**: High model attention (features most influential to prediction)
- **Blue**: Low model attention

Grad-CAM can be enabled from the sidebar in the Streamlit app. Explore standalone in [gradcam.ipynb](gradcam.ipynb).

---

## Configuration

All hyperparameters are centralized in [config.py](config.py):

```python
# Model
MODEL_ARCHITECTURE = "resnet50"
INPUT_SIZE         = 224
NUM_CLASSES        = 2

# Training
BATCH_SIZE         = 32
LEARNING_RATE      = 0.001
NUM_EPOCHS         = 10
FINE_TUNE_EPOCHS   = 5
FINE_TUNE_LR       = 1e-5
WEIGHT_DECAY       = 0.01

# Inference
DYSLEXIA_THRESHOLD = 0.5
HIGH_RISK          = 0.5
MODERATE_RISK      = 0.3
MIN_PATCHES        = 5
PATCH_SIZE         = 224
PATCH_STRIDE       = 224

# Data Augmentation
ROTATION_DEGREES   = 10
FLIP_PROBABILITY   = 0.5
```

---

## Linguistic Features

The 8-dimensional feature vector used for language risk scoring:

| # | Feature | Description |
|---|---|---|
| 1 | Spelling error rate | Fraction of words not in English dictionary |
| 2 | Non-word ratio | Fraction of tokens with no phonetic match in dictionary |
| 3 | Phonetic error ratio | Rate of phonetically confusable errors (Double Metaphone) |
| 4 | Repetition score | Rate of consecutive repeated words |
| 5 | Average word length | Mean character length of tokens |
| 6 | Word count (normalized) | Total words ÷ 100 |
| 7 | Rare word ratio | Fraction of words below Wikipedia frequency threshold |
| 8 | GPT-2 Perplexity | DistilGPT2 perplexity score (log-softened) |

**Feature softening** prevents extreme values from dominating:
- Values are capped (e.g., spelling errors capped at 0.5)
- Log transform applied: `log1p(x)`
- Perplexity softened: `log1p(min(perplexity, 500))`

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | None (CPU inference) | NVIDIA RTX 3050 4GB+ |
| CUDA | — | 11.8+ |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB | 10 GB (with dataset) |

---

## License

This project is intended for educational and research purposes.

---

## Author

**Ayush Singh**  
AI/ML project for early dyslexia screening using multimodal deep learning.
