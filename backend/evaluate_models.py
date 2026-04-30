"""
Model evaluation script — Dyslexia Screening System.

Evaluates baseline and fine-tuned checkpoints side-by-side on the test set.

New in v2:
  - Test-Time Augmentation (TTA) — averages 5 augmented predictions per patch
  - Ensemble mode            — combines ResNet-50 + EfficientNet predictions
  - Saves predictions to CSV for notebook visualisation

Usage:
  python evaluate_models.py                        # ResNet-50 only
  python evaluate_models.py --tta                  # ResNet-50 with TTA
  python evaluate_models.py --ensemble             # ResNet-50 + EfficientNet ensemble
  python evaluate_models.py --arch efficientnet    # EfficientNet only
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from backend import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE     = torch.device(config.DEVICE)
MODELS_DIR = Path(config.MODELS_DIR)
TEST_DIR   = Path(config.TEST_DIR)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
_SIZE         = config.IMAGE_SIZE

# ---------------------------------------------------------------------------
# TTA transforms
# ---------------------------------------------------------------------------
_TTA_TRANSFORMS = [
    transforms.Compose([transforms.Resize((_SIZE, _SIZE)), transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    transforms.Compose([transforms.Resize((_SIZE, _SIZE)), transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    transforms.Compose([transforms.Resize((_SIZE, _SIZE)), transforms.RandomRotation((5, 5)),
                        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    transforms.Compose([transforms.Resize((_SIZE, _SIZE)), transforms.RandomRotation((-5, -5)),
                        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    transforms.Compose([transforms.Resize((_SIZE, _SIZE)), transforms.ColorJitter(brightness=0.2),
                        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
]

test_transform = transforms.Compose([
    transforms.Resize((_SIZE, _SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_resnet50() -> nn.Module:
    from typing import cast
    model       = models.resnet50(weights=None)
    in_features = cast(nn.Linear, model.fc).in_features
    model.fc    = nn.Sequential(          # type: ignore[assignment]
        nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 1), nn.Sigmoid(),
    )
    return model


def _build_efficientnet() -> nn.Module:
    from typing import cast
    model       = models.efficientnet_b0(weights=None)
    in_features = cast(nn.Linear, model.classifier[1]).in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1), nn.Sigmoid())
    return model


def load_checkpoint(path: Path, arch: str = "resnet50") -> nn.Module:
    builder = _build_efficientnet if arch == "efficientnet" else _build_resnet50
    model   = builder()
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    logger.info("Loaded %s: %s", arch, path)
    return model


# ---------------------------------------------------------------------------
# Inference — standard
# ---------------------------------------------------------------------------

def run_evaluation(
    model: nn.Module, loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = model(images.to(DEVICE)).squeeze(1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend((probs > config.DYSLEXIA_THRESHOLD).astype(int))
            all_probs.extend(probs)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ---------------------------------------------------------------------------
# Inference — TTA (5 augmented passes per batch)
# ---------------------------------------------------------------------------

def run_evaluation_tta(
    model: nn.Module, dataset: datasets.ImageFolder
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run TTA inference directly on the dataset (bypasses DataLoader transforms)."""
    logger.info("Running TTA evaluation (%d transforms)…", len(_TTA_TRANSFORMS))

    all_labels: List[int]   = []
    all_probs:  List[float] = []

    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            img_path, label = dataset.samples[idx]
            from PIL import Image as PILImage
            pil_img = PILImage.open(img_path).convert("RGB")

            patch_probs = []
            for t in _TTA_TRANSFORMS:
                tensor: torch.Tensor = t(pil_img)  # type: ignore[assignment]
                patch_probs.append(model(tensor.unsqueeze(0).to(DEVICE)).item())

            all_labels.append(label)
            all_probs.append(float(np.mean(patch_probs)))

            if (idx + 1) % 5000 == 0:
                logger.info("  TTA progress: %d / %d", idx + 1, len(dataset))

    y_true  = np.array(all_labels)
    y_prob  = np.array(all_probs)
    y_pred  = (y_prob > config.DYSLEXIA_THRESHOLD).astype(int)
    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def run_evaluation_ensemble(
    models_list: List[nn.Module], loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average predictions from multiple models."""
    logger.info("Running ensemble evaluation (%d models)…", len(models_list))
    all_labels: List[int]   = []
    all_probs:  List[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            batch_probs = np.zeros(len(labels))
            for m in models_list:
                batch_probs += m(images).squeeze(1).cpu().numpy()
            batch_probs /= len(models_list)
            all_labels.extend(labels.numpy())
            all_probs.extend(batch_probs.tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob > config.DYSLEXIA_THRESHOLD).astype(int)
    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    # y_true / y_pred use ImageFolder labels: dyslexic=0, normal=1
    # y_prob = P(normal) from model sigmoid output
    # AUC: pass P(normal) with pos_label=1 (normal) — symmetric so value is correct
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    # confusion_matrix rows/cols ordered by label: 0=dyslexic, 1=normal
    # [[pred0∩true0, pred1∩true0], [pred0∩true1, pred1∩true1]]
    # = [[correctly_dyslexic, missed_dyslexic], [false_alarm, correctly_normal]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    correctly_dys, missed_dys, false_alarm, correctly_nor = cm.ravel()

    # Dyslexic (label=0) as the target positive class
    prec_d = correctly_dys / max(correctly_dys + false_alarm, 1)
    rec_d  = correctly_dys / max(correctly_dys + missed_dys,  1)
    f1_d   = 2 * prec_d * rec_d / max(prec_d + rec_d, 1e-9)

    # Normal (label=1)
    prec_n = correctly_nor / max(correctly_nor + missed_dys, 1)
    rec_n  = correctly_nor / max(correctly_nor + false_alarm, 1)
    f1_n   = 2 * prec_n * rec_n / max(prec_n + rec_n, 1e-9)

    return dict(
        accuracy=acc, roc_auc=auc,
        precision_dyslexic=prec_d, recall_dyslexic=rec_d, f1_dyslexic=f1_d,
        precision_normal=prec_n,   recall_normal=rec_n,   f1_normal=f1_n,
        correctly_dys=int(correctly_dys), missed_dys=int(missed_dys),
        false_alarm=int(false_alarm),     correctly_nor=int(correctly_nor),
    )


def _bar(v: float, w: int = 28) -> str:
    f = int(round(v * w))
    return "[" + "#" * f + "-" * (w - f) + f"]  {v:.1%}"


def print_report(name: str, m: Dict) -> None:
    print(f"\n  {name}")
    print(f"  Accuracy  : {_bar(m['accuracy'])}")
    print(f"  ROC-AUC   : {_bar(m['roc_auc'])}")
    print(f"\n  Dyslexic class (label=0, target positive):")
    print(f"    Precision : {m['precision_dyslexic']:.4f}")
    print(f"    Recall    : {m['recall_dyslexic']:.4f}   ← higher = fewer missed dyslexic cases")
    print(f"    F1        : {m['f1_dyslexic']:.4f}")
    print(f"\n  Normal class (label=1):")
    print(f"    Precision : {m['precision_normal']:.4f}")
    print(f"    Recall    : {m['recall_normal']:.4f}")
    print(f"    F1        : {m['f1_normal']:.4f}")
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    print(f"                      Pred Dyslexic  Pred Normal")
    print(f"  Actual Dyslexic      {m['correctly_dys']:>7,}       {m['missed_dys']:>7,}")
    print(f"  Actual Normal        {m['false_alarm']:>7,}       {m['correctly_nor']:>7,}")


def print_comparison(results: Dict[str, Dict]) -> None:
    print("\n  HEAD-TO-HEAD COMPARISON")
    keys   = list(results.keys())
    header = f"  {'Metric':<28}" + "".join(f"  {k:>14}" for k in keys)
    print(header)
    print("  " + "-" * (28 + 16 * len(keys)))

    metric_labels = [
        ("Accuracy",           "accuracy"),
        ("ROC-AUC",            "roc_auc"),
        ("Dyslexic Precision", "precision_dyslexic"),
        ("Dyslexic Recall",    "recall_dyslexic"),
        ("Dyslexic F1",        "f1_dyslexic"),
        ("Normal Recall",      "recall_normal"),
    ]
    for label, key in metric_labels:
        vals   = [results[k][key] for k in keys]
        best   = max(vals)
        row    = f"  {label:<28}"
        for v in vals:
            mark = " ✓" if v == best else "  "
            row += f"  {v:>12.4f}{mark}"
        print(row)

    print()
    best_key = max(results, key=lambda k: results[k]["recall_dyslexic"])
    print(f"  Recommended for screening (highest dyslexic recall): {best_key}")
    print()


# ---------------------------------------------------------------------------
# Save predictions CSV
# ---------------------------------------------------------------------------

def save_predictions_csv(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> None:
    out = MODELS_DIR / f"predictions_{name.replace(' ', '_').lower()}.csv"
    # true_label / predicted_label: 0=dyslexic, 1=normal
    # dyslexia_probability = 1 - model_output = P(dyslexic)  (high → more likely dyslexic)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "predicted_label", "dyslexia_probability"])
        for t, p, pr in zip(y_true, y_pred, y_prob):
            writer.writerow([int(t), int(p), f"{1.0 - pr:.6f}"])
    logger.info("Predictions saved → %s", out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',     choices=['resnet50', 'efficientnet'], default='resnet50')
    parser.add_argument('--tta',      action='store_true', help="Use Test-Time Augmentation")
    parser.add_argument('--ensemble', action='store_true', help="Ensemble ResNet-50 + EfficientNet")
    args = parser.parse_args()

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    prefix   = 'resnet50' if args.arch == 'resnet50' else 'efficientnet'
    base_p   = MODELS_DIR / f"{prefix}_dyslexia_base.pth"
    fine_p   = MODELS_DIR / f"{prefix}_dyslexia_finetuned.pth"

    for p in [base_p, fine_p]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}\nRun: python train_vision.py --arch {args.arch}")

    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=test_transform)
    test_loader  = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    logger.info("Test samples: %d | Device: %s | TTA: %s | Ensemble: %s",
                len(test_dataset), DEVICE, args.tta, args.ensemble)

    all_results: Dict[str, Dict] = {}

    # ── Evaluate baseline ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    baseline_model = load_checkpoint(base_p, args.arch)
    if args.tta:
        y_true, y_pred_b, y_prob_b = run_evaluation_tta(baseline_model, test_dataset)
    else:
        y_true, y_pred_b, y_prob_b = run_evaluation(baseline_model, test_loader)
    m_base = compute_metrics(y_true, y_pred_b, y_prob_b)
    tag_b  = f"Baseline ({args.arch}" + ("+TTA" if args.tta else "") + ")"
    print_report(tag_b, m_base)
    save_predictions_csv("baseline", y_true, y_pred_b, y_prob_b)
    all_results[tag_b] = m_base
    del baseline_model; torch.cuda.empty_cache()

    # ── Evaluate fine-tuned ──────────────────────────────────────────────────
    print("=" * 60)
    finetuned_model = load_checkpoint(fine_p, args.arch)
    if args.tta:
        _, y_pred_f, y_prob_f = run_evaluation_tta(finetuned_model, test_dataset)
    else:
        _, y_pred_f, y_prob_f = run_evaluation(finetuned_model, test_loader)
    m_fine = compute_metrics(y_true, y_pred_f, y_prob_f)
    tag_f  = f"Fine-tuned ({args.arch}" + ("+TTA" if args.tta else "") + ")"
    print_report(tag_f, m_fine)
    save_predictions_csv("finetuned", y_true, y_pred_f, y_prob_f)
    all_results[tag_f] = m_fine

    # ── Ensemble (ResNet-50 + EfficientNet) ──────────────────────────────────
    if args.ensemble:
        eff_p = MODELS_DIR / "efficientnet_dyslexia_finetuned.pth"
        if eff_p.exists():
            print("=" * 60)
            eff_model  = load_checkpoint(eff_p, "efficientnet")
            ens_models = [finetuned_model, eff_model]
            _, y_pred_e, y_prob_e = run_evaluation_ensemble(ens_models, test_loader)
            m_ens = compute_metrics(y_true, y_pred_e, y_prob_e)
            tag_e = "Ensemble (ResNet-50 + EfficientNet)"
            print_report(tag_e, m_ens)
            save_predictions_csv("ensemble", y_true, y_pred_e, y_prob_e)
            all_results[tag_e] = m_ens
        else:
            logger.warning("EfficientNet checkpoint not found — skipping ensemble.")

    del finetuned_model; torch.cuda.empty_cache()

    # ── Comparison table ─────────────────────────────────────────────────────
    print("=" * 60)
    print_comparison(all_results)
    print("Predictions CSV files saved to:", MODELS_DIR)
    print("Run notebooks/02_model_results.ipynb to visualise results.")


if __name__ == "__main__":
    main()
