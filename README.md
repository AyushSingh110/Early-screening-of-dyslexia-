---
title: Early Dyslexia Screening
sdk: docker
app_port: 8501
---

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
- [Deployment](#deployment)

---

## Overview

Dyslexia is one of the most common learning disabilities, affecting reading, writing, and spelling abilities. Early identification is critical for timely intervention. This system automates the screening process by analyzing two complementary signals:

1. **Handwriting patterns** — using a fine-tuned ResNet-50 convolutional neural network trained on 128,902 handwriting images to classify patches as dyslexic or typical.
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
   │       Score          │              │  (Rule-based)        │
   └──────────┬──────────┘              └───────────┬──────────┘
              │                                     │
              └──────────────┬──────────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │  Multimodal Fusion   │
                  │  60% vision + 25%   │
                  │  language + 15%     │
                  │  differential       │
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
- **Two-stage transfer learning** — ResNet-50 pretrained on ImageNet; Stage 1 trains the classifier head (10 epochs), Stage 2 unfreezes the last residual block for fine-tuning (5 epochs).
- **Linguistic analysis** — Detects spelling errors, non-words, phonetic confusions, and rare words using PyEnchant, Double Metaphone, and handcrafted NLP features.
- **LLM-based perplexity** — Uses DistilGPT2 to measure text perplexity as a proxy for linguistic fluency.
- **Grad-CAM explainability** — Visualizes model attention on handwriting patches so results are interpretable.
- **Multimodal fusion** — Combines handwriting and language risks with weighted formula for a more robust prediction.
- **Interactive web app** — Streamlit-based UI with a clean, card-style dashboard requiring no technical expertise to operate.

---

## Tech Stack

### Deep Learning & Machine Learning
| Library | Version | Purpose |
|---|---|---|
| PyTorch | ≥ 2.0.0 | Deep learning framework |
| TorchVision | ≥ 0.15.0 | ResNet-50, image transforms |
| Scikit-learn | ≥ 1.3.0 | Metrics, evaluation utilities |
| HuggingFace Transformers | ≥ 4.35.0 | DistilGPT2 for text perplexity |

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
| Metaphone | Double Metaphone phonetic similarity matching |

### Data & Utilities
| Library | Purpose |
|---|---|
| NumPy | Numerical computing and array operations |
| Pandas | CSV dataset loading and manipulation |
| Joblib | Model serialization |
| python-dotenv | Environment variable management |
| kaggle | Kaggle API for dataset download |

### Web Application
| Library | Purpose |
|---|---|
| Streamlit | Interactive web UI |

---

## Project Structure

```
Early-screening-of-dyslexia-/
│
├── frontend/
│   └── app.py                      # Streamlit web application (main entry point)
├── backend/
│   ├── config.py                   # Centralized configuration (hyperparameters, paths, thresholds)
│   ├── train_vision.py             # Two-stage vision model training script
│   ├── evaluate_models.py          # Side-by-side baseline vs fine-tuned evaluation
│   ├── utils/                      # Vision/OCR/report/history utilities
│   └── language_model/             # Linguistic analysis module
├── requirements.txt                # Python dependencies
├── requirements-deploy.txt         # Lean runtime dependencies for Docker deployment
├── Dockerfile                      # Hugging Face Spaces Docker deployment
├── DEPLOYMENT.md                   # Step-by-step deployment guide
├── student_writing.csv             # Linguistic dataset (token-level error annotations)
│
├── models/                         # Saved model weights (not included in repo)
│   ├── resnet50_dyslexia_base.pth       # Stage 1 baseline checkpoint
│   └── resnet50_dyslexia_finetuned.pth  # Stage 2 fine-tuned checkpoint (used for inference)
│
└── data/                           # Dataset directory (not included in repo)
    ├── train/dyslexic/ & normal/   # 128,902 training images
    ├── val/dyslexic/ & normal/     # 22,747 validation images
    └── test/dyslexic/ & normal/    # 56,693 test images
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

# 3. Install dependencies
pip install -r requirements.txt
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
streamlit run frontend/app.py
```

Opens at `http://localhost:8501` in your browser.

**Workflow:**
1. Upload a handwriting image (PNG, JPG, JPEG — max 10 MB)
2. Toggle Grad-CAM visualization from the sidebar (optional)
3. Click **Analyse Handwriting**
4. View:
   - Overall risk level (LOW / MODERATE / HIGH) with color-coded banner
   - Handwriting risk, language risk, and final fused score
   - Per-patch predictions and confidence distribution
   - Grad-CAM heatmaps highlighting model attention regions

### Training

```bash
# Train both baseline and fine-tuned checkpoints:
python -m backend.train_vision

# Evaluate both models side-by-side on the test set:
python -m backend.evaluate_models
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
Adaptive Average Pooling → 2048-dim feature vector
  ↓
FC(2048 → 256) + ReLU + Dropout(0.5)
  ↓
FC(256 → 1) + Sigmoid
  ↓
Output: Dyslexia probability ∈ [0, 1]
```

**Transfer Learning Strategy:**

| Stage | Epochs | Learning Rate | Unfrozen Layers | Saved As |
|---|---|---|---|---|
| 1 — Baseline  | 10 | 1e-3 | Classifier head only | `resnet50_dyslexia_base.pth` |
| 2 — Fine-tune | 5  | 1e-5 | ResNet `layer4` + head | `resnet50_dyslexia_finetuned.pth` |

- **Loss**: Binary Cross-Entropy (BCELoss)
- **Optimizer**: Adam with weight decay 0.01
- **LR schedule**: StepLR (step=3, γ=0.5) during baseline stage
- **Normalization**: ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`

### Language Risk Model

**Pipeline:**
```
Raw OCR Text
  ↓
Tokenization → word list
  ↓
PyEnchant spell-check  →  non_word_ratio        (weight 0.40)
Metaphone matching     →  phonetic_error_ratio  (weight 0.25)
Rare-word detection    →  rare_word_ratio       (weight 0.10)
DistilGPT2             →  perplexity (normalised)(weight 0.25)
  ↓
Weighted sum → language risk ∈ [0, 1]
```

### Multimodal Fusion

```
final_risk = 0.60 × handwriting_risk
           + 0.25 × language_risk
           + 0.15 × max(language_risk − handwriting_risk, 0)
```

- If OCR extracts fewer than 20 words: language component is skipped.
- Safety cap: if `handwriting_risk < 0.2`, `final_risk` is capped at 0.45.

**Risk Thresholds:**

| Level | Score Range |
|---|---|
| LOW | < 0.30 |
| MODERATE | 0.30 – 0.50 |
| HIGH | > 0.50 |

---

## Dataset

- **Source**: [Dr. Iza Sazanita Isa — Dyslexia Handwriting Dataset (Kaggle)](https://www.kaggle.com/datasets/drizasazanitaisa/dyslexia-handwriting-dataset)
- **Original classes**: Normal · Reversal · Corrected
- **Binary mapping**: Reversal + Corrected → `dyslexic`, Normal → `normal`
- **Total images**: 208,342

| Split | Images | Classes |
|---|---|---|
| Train | 128,902 | dyslexic / normal |
| Val   | 22,747  | dyslexic / normal |
| Test  | 56,693  | dyslexic / normal |

**Data augmentation (training only):**
- Random rotation ±10°
- Random horizontal flip 50%
- Color jitter (brightness/contrast ±0.2)
- Resize to 224×224 + ImageNet normalization

---

## Training

```bash
python -m backend.train_vision
```

Runs automatically in two stages and saves both checkpoints to `models/`.

**Expected training time** (NVIDIA RTX 3050 4GB):
- Stage 1 (10 epochs × ~4,000 batches): ~5–6 hours
- Stage 2 (5 epochs fine-tuning): ~2.5–3 hours

---

## Evaluation

```bash
python -m backend.evaluate_models
```

Results on **56,693 test patches** (label 0 = dyslexic, label 1 = normal — ImageFolder alphabetical):

| Metric | Baseline | Fine-tuned | Winner |
|---|---|---|---|
| Accuracy | 68.9% | **77.0%** | Fine-tuned |
| ROC-AUC | 75.7% | **86.0%** | Fine-tuned |
| Dyslexic Precision | 80.3% | **96.6%** | Fine-tuned |
| Dyslexic Recall | **69.5%** | 67.2% | Baseline |
| Dyslexic F1 | 74.5% | **79.3%** | Fine-tuned ✓ |
| Normal Recall | 67.6% | **95.5%** | Fine-tuned |

> **Note**: The fine-tuned model has very high dyslexic precision (96.6%) — when it flags a patch as dyslexic it is almost always correct. Dyslexic recall (67.2%) means ~33% of dyslexic patches are missed at threshold 0.5; lowering the threshold trades precision for recall. For screening, a lower threshold (e.g. 0.35) is recommended to reduce missed cases.

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

Enable from the sidebar in the Streamlit app.

---

## Configuration

All hyperparameters are centralized in [backend/config.py](backend/config.py):

```python
IMAGE_SIZE          = 224       # Input patch size
BATCH_SIZE          = 32
LEARNING_RATE       = 0.001     # Stage 1 LR
EPOCHS              = 10        # Stage 1 epochs
WEIGHT_DECAY        = 0.01

DYSLEXIA_THRESHOLD  = 0.5
HIGH_RISK_THRESHOLD = 0.5
MODERATE_RISK_THRESHOLD = 0.3

PATCH_SIZE          = 224
PATCH_STRIDE        = 224       # No overlap
PATCH_MIN_VARIANCE  = 100.0     # Filters blank/white patches
NUM_WORKERS         = 0         # Windows compatibility
```

---

## Deployment

### Recommended: Hugging Face Spaces Docker

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) and choose **Docker**.
2. Push this repository to the Space.
3. Upload `models/resnet50_dyslexia_finetuned.pth` with Git LFS.
4. The app runs from `frontend/app.py` on port `8501`.

See [DEPLOYMENT.md](DEPLOYMENT.md) for exact commands.

### Alternative: Streamlit Community Cloud

1. Push this repository to GitHub (add model files via Git LFS or host on HuggingFace Hub)
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo and select `frontend/app.py` as the entry point
4. Add any secrets (e.g. Kaggle credentials) in the Secrets panel

### Older Hugging Face Spaces Notes

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Push the repo — HuggingFace supports large model files natively
3. The app runs at `https://huggingface.co/spaces/<username>/<space-name>`

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

Apache 2.0

---
