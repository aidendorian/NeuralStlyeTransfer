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

def visualize_simple(generated, content=None, style=None, save_path=None, normalized=False):
    """Simple visualization of generated image
    
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

def visualize_progress(images_list, losses_list=None, save_path=None, normalized=False):
    """Visualize progress of style transfer over iterations
    
    Args:
        images_list: List of generated image tensors at different iterations
        losses_list: Optional list of (content_loss, style_loss) tuples
        save_path: Optional path to save the figure
        normalized: Set True if images are normalized with ImageNet stats
    """
    num_images = len(images_list)
    
    if losses_list:
        fig = plt.figure(figsize=(4*num_images, 8))
        gs = fig.add_gridspec(2, num_images, height_ratios=[3, 1])
        
        for idx, img in enumerate(images_list):
            ax = fig.add_subplot(gs[0, idx])
            ax.imshow(tensor_to_image(img, normalized))
            if idx == 0:
                ax.set_title('Initial', fontsize=12, fontweight='bold')
            elif idx == num_images - 1:
                ax.set_title('Final', fontsize=12, fontweight='bold')
            else:
                ax.set_title(f'Step {idx}', fontsize=12)
            ax.axis('off')
        
        ax_loss = fig.add_subplot(gs[1, :])
        content_losses = [l[0] for l in losses_list]
        style_losses = [l[1] for l in losses_list]
        iterations = list(range(len(losses_list)))
        
        ax_loss.plot(iterations, content_losses, 'b-', label='Content Loss', linewidth=2)
        ax_loss.plot(iterations, style_losses, 'r-', label='Style Loss', linewidth=2)
        ax_loss.set_xlabel('Iteration', fontsize=12)
        ax_loss.set_ylabel('Loss', fontsize=12)
        ax_loss.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax_loss.legend(fontsize=10)
        ax_loss.grid(True, alpha=0.3)
        
    else:
        fig, axes = plt.subplots(1, num_images, figsize=(4*num_images, 4))
        if num_images == 1:
            axes = [axes]
        
        for idx, img in enumerate(images_list):
            axes[idx].imshow(tensor_to_image(img, normalized))
            if idx == 0:
                axes[idx].set_title('Initial', fontsize=12, fontweight='bold')
            elif idx == num_images - 1:
                axes[idx].set_title('Final', fontsize=12, fontweight='bold')
            else:
                axes[idx].set_title(f'Step {idx}', fontsize=12)
            axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()

def save_image(tensor, save_path, normalized=False):
    """Save a single tensor as an image file
    
    Args:
        tensor: Image tensor
        save_path: Path to save the image
        normalized: Set True if image is normalized with ImageNet stats
    """
    from PIL import Image
    
    img = tensor_to_image(tensor, normalized)
    img = (img * 255).astype(np.uint8)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(save_path)
    print(f"Saved to {save_path}")