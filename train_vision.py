import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import config


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Constants
DEVICE      = torch.device(config.DEVICE)
MODELS_DIR  = Path(config.MODELS_DIR)
TRAIN_DIR   = Path(config.TRAIN_DIR)
VAL_DIR     = Path(config.VAL_DIR)


# Data transforms
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

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


# Helper: build model
def build_model() -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    return model.to(DEVICE)



# Helper: freeze / unfreeze layers
def freeze_backbone(model: nn.Module) -> None:
    """Freeze everything except the custom fc head."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Backbone frozen  — trainable parameters: %s", f"{trainable:,}")


def unfreeze_layer4(model: nn.Module) -> None:
    """Unfreeze layer4 + fc for fine-tuning."""
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Layer4 + fc unfrozen — trainable parameters: %s", f"{trainable:,}")


# Helper: one training epoch
def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    epoch:      int,
    total_epochs: int,
) -> float:
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
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

# Helper: validation
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple:
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()

            preds   = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

# Stage 1 — Baseline training
def train_baseline(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
    logger.info("=" * 60)
    logger.info("STAGE 1 — Baseline training (head only)")
    logger.info("=" * 60)

    freeze_backbone(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, config.EPOCHS)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.2f%% | %.0fs",
            epoch, config.EPOCHS, train_loss, val_loss, val_acc * 100, elapsed,
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = MODELS_DIR / "resnet50_dyslexia_base.pth"
            torch.save(model.state_dict(), save_path)
            logger.info("  -> New best saved: %s  (val_acc=%.2f%%)", save_path, val_acc * 100)

    logger.info("Baseline training complete. Best val accuracy: %.2f%%", best_val_acc * 100)
    return model

# Stage 2 — Fine-tuning
def finetune(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
    logger.info("=" * 60)
    logger.info("STAGE 2 — Fine-tuning (layer4 + head)")
    logger.info("=" * 60)

    # Load best baseline weights before fine-tuning
    base_path = MODELS_DIR / "resnet50_dyslexia_base.pth"
    model.load_state_dict(torch.load(base_path, map_location=DEVICE, weights_only=True))
    logger.info("Loaded baseline checkpoint from %s", base_path)

    unfreeze_layer4(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5,  # Much smaller LR to avoid catastrophic forgetting
        weight_decay=config.WEIGHT_DECAY,
    )
    criterion = nn.BCELoss()

    FINETUNE_EPOCHS = 5
    best_val_acc    = 0.0

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, FINETUNE_EPOCHS)
        val_loss, val_acc = validate(model, val_loader, criterion)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.2f%% | %.0fs",
            epoch, FINETUNE_EPOCHS, train_loss, val_loss, val_acc * 100, elapsed,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = MODELS_DIR / "resnet50_dyslexia_finetuned.pth"
            torch.save(model.state_dict(), save_path)
            logger.info("  -> New best saved: %s  (val_acc=%.2f%%)", save_path, val_acc * 100)

    logger.info("Fine-tuning complete. Best val accuracy: %.2f%%", best_val_acc * 100)
    return model



# Entry point
def main() -> None:
    # Validate data directories
    for d, name in [(TRAIN_DIR, "train"), (VAL_DIR, "val")]:
        if not d.exists():
            raise FileNotFoundError(
                f"{name} directory not found: {d}\n"
                "Expected structure:\n"
                "  data/train/dyslexic/\n"
                "  data/train/normal/\n"
                "  data/val/dyslexic/\n"
                "  data/val/normal/"
            )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    val_dataset   = datasets.ImageFolder(str(VAL_DIR),   transform=val_transform)

    logger.info("Train samples : %d  (%s)", len(train_dataset), train_dataset.class_to_idx)
    logger.info("Val   samples : %d", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    logger.info("Device: %s", DEVICE)

    model = build_model()

    # Stage 1
    model = train_baseline(model, train_loader, val_loader)

    # Stage 2
    model = finetune(model, train_loader, val_loader)

    logger.info("All done. Checkpoints saved to: %s", MODELS_DIR)
    logger.info("  Baseline  : %s", MODELS_DIR / "resnet50_dyslexia_base.pth")
    logger.info("  Fine-tuned: %s", MODELS_DIR / "resnet50_dyslexia_finetuned.pth")
    logger.info("Run evaluate_models.py to compare them.")


if __name__ == "__main__":
    main()
