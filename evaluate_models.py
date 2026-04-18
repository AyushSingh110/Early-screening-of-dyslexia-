import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import config


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Constants
DEVICE     = torch.device(config.DEVICE)
MODELS_DIR = Path(config.MODELS_DIR)
TEST_DIR   = Path(config.TEST_DIR)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

CLASS_NAMES = ["Normal", "Dyslexic"]



# Model builder (mirrors train_vision.py)
def _build_model() -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    return model


def load_checkpoint(path: Path) -> nn.Module:
    model = _build_model()
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    logger.info("Loaded: %s", path)
    return model



# Inference on full test set
def run_evaluation(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (true_labels, predicted_labels, predicted_probabilities).
    """
    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            probs  = model(images).squeeze(1).cpu().numpy()
            preds  = (probs > config.DYSLEXIA_THRESHOLD).astype(int)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )



# Pretty-printing helpers
def _bar(value: float, width: int = 30) -> str:
    filled = int(round(value * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"]  {value:.1%}"


def _confusion_matrix_str(cm: np.ndarray) -> str:
    tn, fp, fn, tp = cm.ravel()
    lines = [
        "                  Predicted Normal   Predicted Dyslexic",
        f"  Actual Normal        {tn:>7,}            {fp:>7,}",
        f"  Actual Dyslexic      {fn:>7,}            {tp:>7,}",
    ]
    return "\n".join(lines)


def print_report(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    acc      = accuracy_score(y_true, y_pred)
    roc_auc  = roc_auc_score(y_true, y_prob)
    cm       = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision_dyslexic = tp / max(tp + fp, 1)
    recall_dyslexic    = tp / max(tp + fn, 1)
    f1_dyslexic        = (
        2 * precision_dyslexic * recall_dyslexic
        / max(precision_dyslexic + recall_dyslexic, 1e-9)
    )

    precision_normal = tn / max(tn + fn, 1)
    recall_normal    = tn / max(tn + fp, 1)
    f1_normal        = (
        2 * precision_normal * recall_normal
        / max(precision_normal + recall_normal, 1e-9)
    )
    print(f"  {name}")
    print(f"  Accuracy  :  {_bar(acc)}")
    print(f"  ROC-AUC   :  {_bar(roc_auc)}")
    print()
    print(f"  Dyslexic class (positive):")
    print(f"    Precision : {precision_dyslexic:.4f}")
    print(f"    Recall    : {recall_dyslexic:.4f}   <- higher is better for screening")
    print(f"    F1        : {f1_dyslexic:.4f}")
    print()
    print(f"  Normal class (negative):")
    print(f"    Precision : {precision_normal:.4f}")
    print(f"    Recall    : {recall_normal:.4f}")
    print(f"    F1        : {f1_normal:.4f}")
    print()
    print("  Confusion matrix:")
    print(_confusion_matrix_str(cm))
    print()

    return {
        "accuracy":            acc,
        "roc_auc":             roc_auc,
        "precision_dyslexic":  precision_dyslexic,
        "recall_dyslexic":     recall_dyslexic,
        "f1_dyslexic":         f1_dyslexic,
        "precision_normal":    precision_normal,
        "recall_normal":       recall_normal,
        "f1_normal":           f1_normal,
    }


def print_comparison(baseline: Dict, finetuned: Dict) -> None:
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"  {'Metric':<28} {'Baseline':>10}  {'Fine-tuned':>10}  {'Winner':>10}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*10}")

    metrics = [
        ("Accuracy",           "accuracy"),
        ("ROC-AUC",            "roc_auc"),
        ("Dyslexic Precision", "precision_dyslexic"),
        ("Dyslexic Recall",    "recall_dyslexic"),
        ("Dyslexic F1",        "f1_dyslexic"),
        ("Normal Precision",   "precision_normal"),
        ("Normal Recall",      "recall_normal"),
        ("Normal F1",          "f1_normal"),
    ]

    for label, key in metrics:
        b = baseline[key]
        f = finetuned[key]
        winner = "Fine-tuned" if f > b else ("Baseline" if b > f else "Tie")
        print(f"  {label:<28} {b:>10.4f}  {f:>10.4f}  {winner:>10}")

    # Overall recommendation
    finetuned_wins = sum(
        1 for _, k in metrics if finetuned[k] > baseline[k]
    )
    print(f"\n  Fine-tuned wins {finetuned_wins}/{len(metrics)} metrics.")
    if finetuned["recall_dyslexic"] >= baseline["recall_dyslexic"]:
        print("  Fine-tuned maintains or improves dyslexic RECALL — recommended for screening.")
    else:
        print(
            "  WARNING: Fine-tuned has LOWER dyslexic recall than baseline.\n"
            "  For a screening tool, baseline may be the safer choice."
        )
    print()

# Entry point
def main() -> None:
    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}\n"
            "Expected:\n"
            "  data/test/dyslexic/\n"
            "  data/test/normal/"
        )

    baseline_path  = MODELS_DIR / "resnet50_dyslexia_base.pth"
    finetuned_path = MODELS_DIR / "resnet50_dyslexia_finetuned.pth"

    missing = [p for p in [baseline_path, finetuned_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model checkpoint(s):\n" +
            "\n".join(f"  {p}" for p in missing) +
            "\nRun `python train_vision.py` first."
        )

    # Data loader
    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=test_transform)
    logger.info("Test samples: %d  (classes: %s)", len(test_dataset), test_dataset.class_to_idx)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # Evaluate baseline
    logger.info("Evaluating baseline model …")
    baseline_model = load_checkpoint(baseline_path)
    y_true, y_pred_base, y_prob_base = run_evaluation(baseline_model, test_loader)
    baseline_metrics = print_report("BASELINE MODEL", y_true, y_pred_base, y_prob_base)
    del baseline_model
    torch.cuda.empty_cache()

    # Evaluate fine-tuned
    logger.info("Evaluating fine-tuned model …")
    finetuned_model = load_checkpoint(finetuned_path)
    _, y_pred_fine, y_prob_fine = run_evaluation(finetuned_model, test_loader)
    finetuned_metrics = print_report("FINE-TUNED MODEL", y_true, y_pred_fine, y_prob_fine)
    del finetuned_model
    torch.cuda.empty_cache()

    # Side-by-side comparison
    print_comparison(baseline_metrics, finetuned_metrics)


if __name__ == "__main__":
    main()
