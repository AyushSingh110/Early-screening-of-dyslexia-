"""
Model loading utilities for the dyslexia screening vision model.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

import config

logger = logging.getLogger(__name__)


def _build_model() -> nn.Module:
    """Construct ResNet-50 with custom binary classification head."""
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    return model


def load_model(model_path: Optional[str] = None) -> nn.Module:
    """
    Load the trained ResNet-50 dyslexia classifier.

    Tries the fine-tuned checkpoint first, falls back to the baseline
    checkpoint if the fine-tuned file is not found.
    """
    device = torch.device(config.DEVICE)

    # Resolve checkpoint path
    if model_path is None:
        finetuned = Path(config.FINETUNED_MODEL_PATH)
        baseline  = Path(config.BASE_MODEL_PATH)

        if finetuned.exists():
            model_path = str(finetuned)
            logger.info("Loading fine-tuned model: %s", model_path)
        elif baseline.exists():
            model_path = str(baseline)
            logger.warning(
                "Fine-tuned model not found; loading baseline model: %s", model_path
            )
        else:
            raise FileNotFoundError(
                f"No model checkpoint found. Expected:\n"
                f"  Fine-tuned : {finetuned}\n"
                f"  Baseline   : {baseline}\n"
                "Train the model first (see train.ipynb)."
            )

    model = _build_model()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Model loaded from %s on %s", model_path, device)
    return model
