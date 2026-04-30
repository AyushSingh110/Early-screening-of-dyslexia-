"""
Model loading and inference utilities for the dyslexia screening vision model.

Includes:
  - ResNet-50 loader (fine-tuned → baseline fallback)
  - EfficientNet-B0 loader
  - Test-Time Augmentation (TTA) for +1-2% accuracy at inference time
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from backend import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ImageNet normalization constants
# ---------------------------------------------------------------------------
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# TTA transform bank
# ---------------------------------------------------------------------------
_SIZE = config.IMAGE_SIZE

_TTA_TRANSFORMS = [
    # 1. Original
    transforms.Compose([
        transforms.Resize((_SIZE, _SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    # 2. Horizontal flip
    transforms.Compose([
        transforms.Resize((_SIZE, _SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    # 3. Slight rotation +5°
    transforms.Compose([
        transforms.Resize((_SIZE, _SIZE)),
        transforms.RandomRotation(degrees=(5, 5)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    # 4. Slight rotation -5°
    transforms.Compose([
        transforms.Resize((_SIZE, _SIZE)),
        transforms.RandomRotation(degrees=(-5, -5)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    # 5. Slight brightness boost
    transforms.Compose([
        transforms.Resize((_SIZE, _SIZE)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
]


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_resnet50() -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    return model


def _build_efficientnet_b0() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )
    return model


# ---------------------------------------------------------------------------
# Public: load model
# ---------------------------------------------------------------------------

def load_model(model_path: Optional[str] = None, arch: str = "resnet50") -> nn.Module:
    """
    Load a trained dyslexia classifier.

    arch       : 'resnet50' (default) or 'efficientnet'
    model_path : explicit .pth path; if None, auto-resolves from config.

    Falls back: fine-tuned → baseline (for resnet50 only).
    """
    device = torch.device(config.DEVICE)

    if model_path is None:
        if arch == "efficientnet":
            eff_path = Path(config.MODELS_DIR) / "efficientnet_dyslexia_finetuned.pth"
            if not eff_path.exists():
                raise FileNotFoundError(
                    f"EfficientNet checkpoint not found: {eff_path}\n"
                    "Train it first with: python train_vision.py --arch efficientnet"
                )
            model_path = str(eff_path)
        else:
            finetuned = Path(config.FINETUNED_MODEL_PATH)
            baseline  = Path(config.BASE_MODEL_PATH)
            if finetuned.exists():
                model_path = str(finetuned)
                logger.info("Loading fine-tuned model: %s", model_path)
            elif baseline.exists():
                model_path = str(baseline)
                logger.warning("Fine-tuned not found; loading baseline: %s", model_path)
            else:
                raise FileNotFoundError(
                    f"No checkpoint found.\n  Fine-tuned: {finetuned}\n  Baseline: {baseline}"
                )

    builder = _build_efficientnet_b0 if arch == "efficientnet" else _build_resnet50
    model   = builder()
    state   = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    logger.info("Model (%s) loaded from %s on %s", arch, model_path, device)
    return model


# ---------------------------------------------------------------------------
# Public: single-pass prediction
# ---------------------------------------------------------------------------

def predict_patch(
    model: nn.Module,
    patch_img: Image.Image,
    device: torch.device,
) -> float:
    """Return dyslexia probability for a single PIL patch."""
    t = _TTA_TRANSFORMS[0]               # standard transform (no augmentation)
    tensor = t(patch_img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor).item()


# ---------------------------------------------------------------------------
# Public: TTA prediction
# ---------------------------------------------------------------------------

def predict_patch_tta(
    model: nn.Module,
    patch_img: Image.Image,
    device: torch.device,
    n_augments: int = 5,
) -> float:
    """
    Test-Time Augmentation: average predictions over multiple augmented views.

    Typically gives +1-2% accuracy over a single forward pass at no training cost.

    Parameters
    ----------
    n_augments : number of TTA transforms to use (max 5, default 5)
    """
    transforms_to_use = _TTA_TRANSFORMS[:min(n_augments, len(_TTA_TRANSFORMS))]
    probs: List[float] = []

    with torch.no_grad():
        for t in transforms_to_use:
            tensor = t(patch_img).unsqueeze(0).to(device)
            probs.append(model(tensor).item())

    return float(np.mean(probs))


# ---------------------------------------------------------------------------
# Public: ensemble prediction
# ---------------------------------------------------------------------------

def predict_patch_ensemble(
    models_list: List[nn.Module],
    patch_img: Image.Image,
    device: torch.device,
    use_tta: bool = True,
) -> float:
    """
    Average predictions from multiple models (ensemble).

    Typically gives +1-3% accuracy over a single model.
    """
    probs: List[float] = []
    fn = predict_patch_tta if use_tta else predict_patch

    for model in models_list:
        probs.append(fn(model, patch_img, device))

    return float(np.mean(probs))
