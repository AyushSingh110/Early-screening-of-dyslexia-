import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GradCAM:

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):

        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
        logger.info(f"GradCAM initialized for layer: {target_layer.__class__.__name__}")
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        
        def forward_hook(module, input, output):
            """Capture forward pass activations."""
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """Capture backward pass gradients."""
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def remove_hooks(self):
        """Remove registered hooks to free memory."""
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        
        # Ensure gradient computation
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # For binary classification with sigmoid output
        if target_class is None:
            # Use the model's prediction
            score = output
        else:
            score = output[:, target_class]
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Check if gradients and activations were captured
        if self.gradients is None or self.activations is None:
            logger.error("Gradients or activations not captured. Check hooks.")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        # Calculate weights using global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # Apply ReLU to focus on positive influences
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            logger.warning("CAM has no positive values")
        
        return cam.cpu().numpy()
    
    def generate_batch(self, input_tensors: torch.Tensor) -> list:
        cams = []
        for i in range(input_tensors.shape[0]):
            cam = self.generate(input_tensors[i:i+1])
            cams.append(cam)
        return cams


def overlay_gradcam(
    image_np: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:

    # Resize CAM to match image dimensions
    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    
    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    
    # Convert heatmap from BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Ensure image is uint8
    if image_np.dtype != np.uint8:
        image_np = np.uint8(image_np)
    
    # Blend original image with heatmap
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    
    return overlay


def generate_gradcam_visualization(
    model: torch.nn.Module,
    image_np: np.ndarray,
    input_tensor: torch.Tensor,
    device: torch.device,
    target_layer_name: str = "layer4"
) -> Tuple[np.ndarray, np.ndarray]:

    try:
        # Get target layer from model
        target_layer = None
        if hasattr(model, 'base_model'):
            # If using custom wrapper
            target_layer = getattr(model.base_model, target_layer_name, None)
        else:
            # Direct ResNet model
            target_layer = getattr(model, target_layer_name, None)
        
        if target_layer is None:
            logger.error(f"Target layer '{target_layer_name}' not found in model")
            return None, None
        
        # Initialize Grad-CAM
        gradcam = GradCAM(model, target_layer)
        
        # Move input to device
        input_tensor = input_tensor.to(device)
        
        # Generate CAM
        cam = gradcam.generate(input_tensor)
        
        # Create overlay
        overlay = overlay_gradcam(image_np, cam, alpha=0.4)
        
        # Clean up hooks
        gradcam.remove_hooks()
        
        return cam, overlay
    
    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {str(e)}")
        return None, None


def create_side_by_side_visualization(
    original: np.ndarray,
    overlay: np.ndarray,
    cam: np.ndarray
) -> np.ndarray:

    # Resize all to same height
    h = original.shape[0]
    w = original.shape[1]
    
    # Create heatmap visualization
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Concatenate horizontally
    combined = np.hstack([original, heatmap, overlay])
    
    return combined


def batch_gradcam_visualization(
    model: torch.nn.Module,
    patches: list,
    input_tensors: torch.Tensor,
    device: torch.device,
    top_k: int = 5
) -> list:

    model.eval()
    
    # Get predictions for all patches
    with torch.no_grad():
        outputs = model(input_tensors.to(device))
        confidences = outputs.squeeze().cpu().numpy()
    
    # Get indices of top-k most confident predictions
    top_indices = np.argsort(confidences)[-top_k:][::-1]
    
    # Generate Grad-CAM for top patches
    visualizations = []
    for idx in top_indices:
        cam, overlay = generate_gradcam_visualization(
            model,
            patches[idx],
            input_tensors[idx:idx+1],
            device
        )
        if overlay is not None:
            visualizations.append((idx, confidences[idx], overlay))
    
    return visualizations