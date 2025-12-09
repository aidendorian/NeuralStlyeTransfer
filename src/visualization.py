import torch
import matplotlib.pyplot as plt
from pathlib import Path

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a tensor using standard ImageNet normalization parameters.

    This function reverses the normalization process typically applied to images
    before feeding them into pre-trained neural networks. It scales the tensor
    values back to the [0, 1] range using the provided mean and standard deviation.

    Args:
        tensor (torch.Tensor): A normalized tensor of shape (C, H, W) or (B, C, H, W)
            where C=3 (RGB channels), H is height, and W is width.
        mean (list, optional): Mean values used during normalization. Defaults to
            ImageNet means [0.485, 0.456, 0.406] for R, G, B channels respectively.
        std (list, optional): Standard deviation values used during normalization.
            Defaults to ImageNet standard deviations [0.229, 0.224, 0.225] for R, G, B
            channels respectively.

    Returns:
        torch.Tensor: Denormalized tensor in the [0, 1] range with the same shape
            as the input tensor.
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def tensor_to_image(tensor, normalized=False):
    """
    Convert a PyTorch tensor to a numpy image array.
    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W) or (B, C, H, W).
            If 4D, only the first batch element is used.
        normalized (bool, optional): If True, denormalize the tensor before conversion.
            Defaults to False.
    Returns:
        numpy.ndarray: Image array of shape (H, W, C) with values clamped to [0, 1].
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
    """
    Visualize generated image alongside optional content and style reference images.
    Creates a matplotlib figure displaying the generated image and optionally the content
    and/or style images used for neural style transfer. The layout adjusts dynamically
    based on which images are provided.
    Args:
        generated: Tensor representation of the generated/output image to display.
        content (optional): Tensor representation of the content image. If provided,
            will be displayed on the left. Defaults to None.
        style (optional): Tensor representation of the style image. If provided,
            will be displayed on the right. Defaults to None.
        save_path (optional): File path where the figure should be saved. If provided,
            the figure is saved as an image file before displaying. Parent directories
            are created if they don't exist. Defaults to None.
        normalized (bool): Whether the input tensors are normalized to [0, 1] range.
            If False, assumes tensors are in [0, 255] range. Defaults to False.
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
