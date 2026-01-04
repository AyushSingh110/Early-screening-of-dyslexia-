from torchvision import transforms
from PIL import Image
import torch
from typing import Union
import logging

import config

logger = logging.getLogger(__name__)

# ImageNet normalization statistics (ResNet-50 pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard preprocessing pipeline for inference
transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Preprocessing with additional augmentation (for training/validation)
transform_with_augmentation = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def preprocess_image(
    image: Union[Image.Image, str],
    augment: bool = False
) -> torch.Tensor:
    
    # Load image if path is provided
    if isinstance(image, str):
        try:
            image = Image.open(image).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image from {image}: {str(e)}")
            raise
    
    # Ensure image is RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply appropriate transformation
    if augment:
        tensor = transform_with_augmentation(image)
    else:
        tensor = transform(image)
    
    return tensor

def preprocess_batch(
    images: list,
    augment: bool = False
) -> torch.Tensor:
    
    tensors = []
    
    for img in images:
        tensor = preprocess_image(img, augment=augment)
        tensors.append(tensor)
    
    return torch.stack(tensors)

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    denormalized = tensor * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)
    
    return denormalized

def validate_input_tensor(tensor: torch.Tensor) -> bool:
    # Check dimensions
    if tensor.dim() not in [3, 4]:
        raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")
    
    # Check channels
    if tensor.dim() == 3:
        if tensor.shape[0] != 3:
            raise ValueError(f"Expected 3 channels, got {tensor.shape[0]}")
    elif tensor.dim() == 4:
        if tensor.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {tensor.shape[1]}")
    
    # Check spatial dimensions
    expected_size = config.IMAGE_SIZE
    if tensor.dim() == 3:
        if tensor.shape[1] != expected_size or tensor.shape[2] != expected_size:
            raise ValueError(
                f"Expected spatial size {expected_size}x{expected_size}, "
                f"got {tensor.shape[1]}x{tensor.shape[2]}"
            )
    elif tensor.dim() == 4:
        if tensor.shape[2] != expected_size or tensor.shape[3] != expected_size:
            raise ValueError(
                f"Expected spatial size {expected_size}x{expected_size}, "
                f"got {tensor.shape[2]}x{tensor.shape[3]}"
            )
    
    logger.debug(f"Input tensor validated: shape={tensor.shape}")
    return True