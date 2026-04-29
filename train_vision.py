"""
Vision model training script for the Dyslexia Screening System.

Supports two architectures: ResNet-50 (default) and EfficientNet-B0.

Improvements over v1:
  - WeightedRandomSampler  — fixes class imbalance without architecture changes
  - FocalLoss              — down-weights easy negatives, focuses on hard examples
  - EfficientNet-B0 option — compare against ResNet-50

Stages (both architectures):
  Stage 1 — Baseline : freeze backbone, train head only (10 epochs)
  Stage 2 — Fine-tune: unfreeze last block + head (5 epochs, LR=1e-5)

Usage:
  python train_vision.py                        # ResNet-50 (default)
  python train_vision.py --arch efficientnet    # EfficientNet-B0

Saved checkpoints:
  models/resnet50_dyslexia_base.pth
  models/resnet50_dyslexia_finetuned.pth
  models/efficientnet_dyslexia_base.pth         (if --arch efficientnet)
  models/efficientnet_dyslexia_finetuned.pth
"""

import argparse
import logging
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE     = torch.device(config.DEVICE)
MODELS_DIR = Path(config.MODELS_DIR)
TRAIN_DIR  = Path(config.TRAIN_DIR)
VAL_DIR    = Path(config.VAL_DIR)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
    transforms.RandomRotation(config.RANDOM_ROTATION_DEGREES),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017): down-weights easy examples so the model
    focuses on hard / misclassified ones. Works with sigmoid output.

    FL(p) = -alpha * (1 - p_t)^gamma * log(p_t)
    alpha=0.25, gamma=2 are standard values from the original paper.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt   = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_resnet50() -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = cast(nn.Linear, model.fc).in_features
    model.fc = nn.Sequential(          # type: ignore[assignment]
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    return model.to(DEVICE)


def build_efficientnet() -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = cast(nn.Linear, model.classifier[1]).in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )
    return model.to(DEVICE)


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_backbone(model: nn.Module, arch: str) -> None:
    head_key = 'fc' if arch == 'resnet50' else 'classifier'
    for name, param in model.named_parameters():
        param.requires_grad = head_key in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Backbone frozen — trainable params: %s", f"{trainable:,}")


def unfreeze_last_block(model: nn.Module, arch: str) -> None:
    keys = ('layer4', 'fc') if arch == 'resnet50' else ('features.8', 'classifier')
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in keys)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Last block unfrozen — trainable params: %s", f"{trainable:,}")


# ---------------------------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------------------------

def make_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """Over-sample the minority class → balanced mini-batches."""
    targets      = torch.tensor(dataset.targets)
    class_count  = torch.bincount(targets)
    class_weight = 1.0 / class_count.float()
    sample_weight = class_weight[targets]

    logger.info(
        "Class counts: %s",
        {c: int(class_count[i]) for i, c in enumerate(dataset.classes)},
    )
    return WeightedRandomSampler(
        weights=sample_weight.tolist(), num_samples=len(sample_weight), replacement=True
    )


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs) -> float:
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            logger.info(
                "Epoch [%d/%d]  Batch [%d/%d]  Loss: %.4f",
                epoch, total_epochs, batch_idx + 1, len(loader),
                running_loss / (batch_idx + 1),
            )
    return running_loss / len(loader)


def validate(model, loader, criterion) -> tuple:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct    += ((outputs > 0.5).float() == labels).sum().item()
            total      += labels.size(0)
    return total_loss / len(loader), correct / total


# ---------------------------------------------------------------------------
# Stage 1 — Baseline
# ---------------------------------------------------------------------------

def train_baseline(model, train_loader, val_loader, arch, prefix) -> nn.Module:
    logger.info("=" * 60)
    logger.info("STAGE 1 — Baseline (%s, head only)", arch)
    logger.info("=" * 60)

    freeze_backbone(model, arch)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
    )
    criterion = FocalLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    best_val_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, config.EPOCHS)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        logger.info(
            "Epoch %2d/%d | train=%.4f | val=%.4f | acc=%.2f%% | %.0fs",
            epoch, config.EPOCHS, train_loss, val_loss, val_acc * 100, time.time() - t0,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = MODELS_DIR / f"{prefix}_dyslexia_base.pth"
            torch.save(model.state_dict(), path)
            logger.info("  -> Best saved: %s (%.2f%%)", path, val_acc * 100)

    logger.info("Baseline complete. Best val acc: %.2f%%", best_val_acc * 100)
    return model


# ---------------------------------------------------------------------------
# Stage 2 — Fine-tuning
# ---------------------------------------------------------------------------

def finetune(model, train_loader, val_loader, arch, prefix) -> nn.Module:
    logger.info("=" * 60)
    logger.info("STAGE 2 — Fine-tuning (%s, last block + head)", arch)
    logger.info("=" * 60)

    path = MODELS_DIR / f"{prefix}_dyslexia_base.pth"
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    logger.info("Loaded baseline from %s", path)

    unfreeze_last_block(model, arch)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5, weight_decay=config.WEIGHT_DECAY,
    )
    criterion       = FocalLoss()
    FINETUNE_EPOCHS = 5
    best_val_acc    = 0.0

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, FINETUNE_EPOCHS)
        val_loss, val_acc = validate(model, val_loader, criterion)
        logger.info(
            "Epoch %2d/%d | train=%.4f | val=%.4f | acc=%.2f%% | %.0fs",
            epoch, FINETUNE_EPOCHS, train_loss, val_loss, val_acc * 100, time.time() - t0,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = MODELS_DIR / f"{prefix}_dyslexia_finetuned.pth"
            torch.save(model.state_dict(), path)
            logger.info("  -> Best saved: %s (%.2f%%)", path, val_acc * 100)

    logger.info("Fine-tuning complete. Best val acc: %.2f%%", best_val_acc * 100)
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch', choices=['resnet50', 'efficientnet'], default='resnet50',
        help="Architecture to train (default: resnet50)",
    )
    args   = parser.parse_args()
    arch   = args.arch
    prefix = 'resnet50' if arch == 'resnet50' else 'efficientnet'

    for d, name in [(TRAIN_DIR, "train"), (VAL_DIR, "val")]:
        if not d.exists():
            raise FileNotFoundError(f"{name} directory not found: {d}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    val_dataset   = datasets.ImageFolder(str(VAL_DIR),   transform=val_transform)
    logger.info("Train: %d | Val: %d | Device: %s | Arch: %s",
                len(train_dataset), len(val_dataset), DEVICE, arch.upper())

    sampler      = make_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        sampler=sampler, num_workers=config.NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = build_efficientnet() if arch == 'efficientnet' else build_resnet50()
    model = train_baseline(model, train_loader, val_loader, arch, prefix)
    model = finetune(model, train_loader, val_loader, arch, prefix)

    logger.info("All done. Checkpoints saved to: %s", MODELS_DIR)
    logger.info("Run: python evaluate_models.py --arch %s", arch)


if __name__ == "__main__":
    main()
