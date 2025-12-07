import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a normalized tensor back to [0, 1] range"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def tensor_to_image(tensor, normalized=False):
    """Convert a PyTorch tensor to a numpy image for visualization
    
    Args:
        tensor: (C, H, W) or (B, C, H, W) tensor
        normalized: If True, denormalize using ImageNet stats first
    
    Returns:
        numpy array in range [0, 1] with shape (H, W, C)
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu()
    if normalized:
        img = denormalize(img)
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    
    return img

def visualize(generated, content=None, style=None, save_path=None, normalized=False):
    """Visualization of generated image
    
    Args:
        generated: Generated image tensor
        content: Optional content image tensor
        style: Optional style image tensor
        save_path: Optional path to save the figure
        normalized: Set True if images are normalized with ImageNet stats
    """
    num_images = 1 + (content is not None) + (style is not None)
    fig, axes = plt.subplots(1, num_images, figsize=(6*num_images, 6))
    if num_images == 1:
        axes = [axes]
    idx = 0
    
    if content is not None:
        axes[idx].imshow(tensor_to_image(content, normalized))
        axes[idx].set_title('Content Image', fontsize=14, fontweight='bold')
        axes[idx].axis('off')
        idx += 1
    
    axes[idx].imshow(tensor_to_image(generated, normalized))
    axes[idx].set_title('Generated Image', fontsize=14, fontweight='bold')
    axes[idx].axis('off')
    idx += 1
    
    if style is not None:
        axes[idx].imshow(tensor_to_image(style, normalized))
        axes[idx].set_title('Style Image', fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
