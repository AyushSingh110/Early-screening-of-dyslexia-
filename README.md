<div align="center">

# Dyslexia Early Screening System

**AI-powered multimodal handwriting analysis for early dyslexia detection**

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/AImRs/dyslexia-early-screening-system)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Clinical Disclaimer** — This tool is intended for early screening purposes only and does not constitute a medical diagnosis. All results should be reviewed by a qualified professional.

</div>

---

## Overview

Dyslexia affects an estimated 10–15% of the population, yet early identification remains largely manual and inaccessible. This system automates the screening process by analysing two complementary signals from a single handwriting image:

| Modality | Method | Weight |
|---|---|---|
| **Vision** | Patch-based ResNet-50 classifier (fine-tuned on 208K images) | 60% |
| **Language** | Spell-check + phonetic error detection + DistilGPT2 perplexity | 25% |
| **Differential** | Cross-modal disagreement penalty | 15% |

Results are presented as a **LOW / MODERATE / HIGH** risk score with Grad-CAM visual explanations — no technical knowledge required to interpret.

---

## Key Results

| Metric | Baseline | Fine-tuned |
|---|---|---|
| Accuracy | 68.9% | **77.0%** |
| ROC-AUC | 75.7% | **86.0%** |
| Dyslexic Precision | 80.3% | **96.6%** |
| Dyslexic Recall | 69.5% | 67.2% |
| Dyslexic F1 | 74.5% | **79.3%** |
| Normal Recall | 67.6% | **95.5%** |

Evaluated on **56,693 held-out test patches**. At the default threshold (0.5), the fine-tuned model achieves 96.6% precision on dyslexic patches — nearly eliminating false positives. Lowering the threshold to 0.35 recovers recall for higher-sensitivity screening.

---

## System Architecture

```
                   ┌──────────────────────┐
                   │    Handwriting Image  │
                   └──────────┬───────────┘
                              │
             ┌────────────────┴────────────────┐
             │                                 │
  ┌──────────▼──────────┐          ┌───────────▼──────────┐
  │   Patch Extraction   │          │    Tesseract OCR      │
  │   (224×224, stride)  │          │    Text Extraction    │
  └──────────┬──────────┘          └───────────┬──────────┘
             │                                 │
  ┌──────────▼──────────┐          ┌───────────▼──────────┐
  │  ResNet-50 Classifier│          │  Linguistic Features  │
  │  P(dyslexic) per     │          │  + DistilGPT2        │
  │  patch               │          │  Perplexity          │
  └──────────┬──────────┘          └───────────┬──────────┘
             │                                 │
  ┌──────────▼──────────┐          ┌───────────▼──────────┐
  │  Handwriting Risk    │          │  Language Risk Score  │
  └──────────┬──────────┘          └───────────┬──────────┘
             │                                 │
             └──────────────┬──────────────────┘
                            │
                 ┌──────────▼──────────┐
                 │  Multimodal Fusion   │
                 │  60% · 25% · 15%    │
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

- **Patch-based inference** — Variable-size handwriting pages are tiled into 224×224 patches and analysed independently, then aggregated into a page-level score.
- **Two-stage transfer learning** — Stage 1 trains the classifier head; Stage 2 unfreezes ResNet-50's `layer4` for domain-specific fine-tuning.
- **Linguistic analysis** — Non-word ratio, phonetic error ratio (Double Metaphone), rare-word ratio, and normalised DistilGPT2 perplexity combined via weighted sum.
- **Grad-CAM explainability** — Gradient-weighted Class Activation Maps highlight which handwriting regions most influenced each prediction.
- **Multimodal fusion** — Weighted combination of vision and language risk with a cross-modal differential penalty for robustness when signals disagree.
- **Clinical web interface** — Streamlit app with a dark-sidebar EHR-style layout, risk banners, per-patch confidence distribution, and Grad-CAM overlays.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Deep Learning | PyTorch ≥ 2.0, TorchVision ≥ 0.15 |
| Language Models | HuggingFace Transformers ≥ 4.35 (DistilGPT2) |
| Computer Vision | OpenCV, Pillow |
| OCR | pytesseract (Tesseract 5) |
| NLP | PyEnchant, Metaphone |
| ML Utilities | Scikit-learn, NumPy, Pandas |
| Web App | Streamlit |
| Deployment | Docker, Hugging Face Spaces |

---

## Project Structure

```
Early-screening-of-dyslexia-/
│
├── frontend/
│   └── app.py                       # Streamlit web application
├── backend/
│   ├── config.py                    # Centralised hyperparameters & paths
│   ├── train_vision.py              # Two-stage training script
│   ├── evaluate_models.py           # Baseline vs fine-tuned evaluation
│   ├── utils/                       # Vision, OCR, report, history utilities
│   └── language_model/              # Linguistic analysis module
├── notebooks/
│   ├── 01_dataset_analysis.ipynb    # Dataset EDA
│   ├── 02_model_results.ipynb       # Evaluation plots & metrics
│   └── 03_gradcam_visualization.ipynb
├── scripts/
│   └── deploy_to_hf_space.py        # Hugging Face Spaces deploy script
├── Dockerfile
├── requirements.txt
└── requirements-deploy.txt          # Lean runtime deps for Docker
```

---

## Installation

### Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on the system

| Platform | Install |
|---|---|
| Windows | [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) |
| Linux | `sudo apt-get install tesseract-ocr` |
| macOS | `brew install tesseract` |

### Setup

```bash
git clone https://github.com/AyushSingh110/Early-screening-of-dyslexia-.git
cd Early-screening-of-dyslexia-

conda create -n dyslexia_env python=3.10
conda activate dyslexia_env

pip install -r requirements.txt
```

### Run the App

```bash
streamlit run frontend/app.py
# → http://localhost:8501
```

---

## Usage

1. Upload a handwriting image (PNG / JPG / JPEG, max 10 MB)
2. Optionally enable **Grad-CAM** from the sidebar
3. Click **Analyse Handwriting**
4. Review the risk report — overall level, per-modality scores, patch confidence chart, and heatmaps

---

## Models

### Vision — ResNet-50

```
Input (B, 3, 224, 224)
  → ResNet-50 backbone (ImageNet pretrained)
  → Adaptive Average Pool → 2048-d vector
  → FC(2048→256) + ReLU + Dropout(0.5)
  → FC(256→1) + Sigmoid
  → P(dyslexic) ∈ [0, 1]
```

| Stage | Epochs | LR | Unfrozen |
|---|---|---|---|
| 1 — Baseline | 10 | 1e-3 | Head only |
| 2 — Fine-tune | 5 | 1e-5 | `layer4` + head |

Optimiser: Adam (weight decay 0.01) · Loss: BCELoss · LR schedule: StepLR (step=3, γ=0.5)

### Language Risk

```
non_word_ratio       × 0.40   (PyEnchant spell-check)
phonetic_error_ratio × 0.25   (Double Metaphone)
rare_word_ratio      × 0.10
perplexity_norm      × 0.25   (DistilGPT2)
─────────────────────────────
language_risk ∈ [0, 1]
```

### Multimodal Fusion

```
final_risk = 0.60 × vision_risk
           + 0.25 × language_risk
           + 0.15 × max(language_risk − vision_risk, 0)
```

- Language component skipped if OCR yields < 20 words.
- If `vision_risk < 0.2`, `final_risk` is capped at 0.45.

| Risk Level | Score |
|---|---|
| LOW | < 0.30 |
| MODERATE | 0.30 – 0.50 |
| HIGH | > 0.50 |

---

## Dataset

**Source:** [Dyslexia Handwriting Dataset — Dr. Iza Sazanita Isa (Kaggle)](https://www.kaggle.com/datasets/drizasazanitaisa/dyslexia-handwriting-dataset)

Binary mapping: `Reversal + Corrected → dyslexic`, `Normal → normal`

| Split | Images |
|---|---|
| Train | 128,902 |
| Val | 22,747 |
| Test | 56,693 |
| **Total** | **208,342** |

Training augmentation: random rotation ±10°, horizontal flip 50%, colour jitter, ImageNet normalisation.

---

## Training

```bash
# Train both stages sequentially
python -m backend.train_vision

# Evaluate on test set
python -m backend.evaluate_models
```

Expected time on RTX 3050 4 GB: Stage 1 ~5–6 h, Stage 2 ~2.5–3 h.

---

## Deployment

The app is deployed live on Hugging Face Spaces (Docker):

**[https://huggingface.co/spaces/AImRs/dyslexia-early-screening-system](https://huggingface.co/spaces/AImRs/dyslexia-early-screening-system)**

To redeploy after changes:

```bash
python scripts/deploy_to_hf_space.py
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | None (CPU inference) | NVIDIA RTX 3050 4 GB+ |
| CUDA | — | 11.8+ |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB | 10 GB (with dataset) |

---

## License

MIT
