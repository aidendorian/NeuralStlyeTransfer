import torch
import torch.nn as nn
from torch.nn.functional import l1_loss
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose
import torch.nn.functional as F
from torch.optim import Adam, LBFGS
from tqdm import tqdm
from skimage.exposure import match_histograms

def load_image(path, max_side=1280):
    """
    Load an image from the specified path and convert it to a normalized tensor.
    The image is resized if its longest dimension exceeds max_side, maintaining
    aspect ratio. The image is then converted to a tensor and normalized using
    ImageNet normalization parameters.
    Args:
        path (str): File path to the image to load.
        max_side (int, optional): Maximum length of the longest side of the image.
            If the image's longest side exceeds this value, it will be resized
            proportionally. Defaults to 1280.
    Returns:
        tuple: A tuple containing:
            - tensor (torch.Tensor): The normalized image tensor with shape (1, 3, H, W).
            - size (tuple): The original image dimensions as (width, height).
    """
    img = Image.open(path).convert('RGB')
    w, h = img.size
    
    if max(w, h) > max_side:
        if w > h:
            new_w, new_h = max_side, int(h * max_side / w)
        else:
            new_w, new_h = int(w * max_side / h), max_side
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    transform = Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor, (img.width, img.height)

def run_pyramid(content_img, style_img, style_layers, style_weights, extractor, pyramid_levels, adam_iters, adam_lr, alpha, beta):
    """
    Performs multi-scale neural style transfer using an image pyramid approach.

    Args:
        content_img (torch.Tensor): The content image tensor of shape (N, C, H, W).
        style_img (torch.Tensor): The style image tensor of shape (N, C, H, W).
        style_layers (list of str): List of layer names to use for style representation.
        style_weights (list of float): List of weights for each style layer.
        extractor (callable): Feature extractor that returns a dict with 'content' and 'style' features.
        pyramid_levels (int): Number of pyramid levels (scales) to use.
        adam_iters (int): Total number of Adam optimizer iterations.
        adam_lr (float): Learning rate for the Adam optimizer.
        alpha (float): Weight for the content loss.
        beta (float): Weight for the style loss.

    Returns:
        torch.Tensor: The stylized image tensor of shape (N, C, H, W).
    """
    img = content_img.clone().detach().requires_grad_(True)
    _, _, H, W = img.shape

    for level in range(pyramid_levels):
        scale = 2 ** (pyramid_levels - level - 1)
        h, w = H//scale, W//scale

        c_resized = F.interpolate(content_img, size=(h, w), mode='bilinear', align_corners=False)
        s_resized = F.interpolate(style_img, size=(h, w), mode='bilinear', align_corners=False)
        img_resized = F.interpolate(img, size=(h, w), mode='bilinear', align_corners=False)
        img_resized = img_resized.detach().clone().requires_grad_(True)

        with torch.no_grad():
            content_feat = extractor(c_resized)['content']
            style_feats = extractor(s_resized)['style']

        content_loss_mod = loss_content(content_feat)
        style_loss_mods = {l: loss_style(style_feats[l], w) for l, w in zip(style_layers, style_weights)}

        optimizer = Adam([img_resized], lr=adam_lr)
        iters = adam_iters // pyramid_levels + (30 if level == pyramid_levels-1 else 0)
        pbar = tqdm(range(iters), desc=f'Level {level+1}/{pyramid_levels} → {w}×{h}')

        for _ in pbar:
            optimizer.zero_grad()
            feats = extractor(img_resized)

            content_loss_mod(feats['content'])
            c_loss = content_loss_mod.loss
            s_loss = 0.0
            for l in style_layers:
                style_loss_mods[l](feats['style'][l])
                s_loss += style_loss_mods[l].loss

            total = alpha * c_loss + beta * s_loss
            total.backward()
            optimizer.step()
            img_resized.data.clamp_(-3, 3)
            pbar.set_postfix({'Loss': f'{total.item():.6f}'})

        img = F.interpolate(img_resized.detach(), size=(H, W), mode='bilinear', align_corners=False).requires_grad_(True)

    return img

def lbfgs_polish(content_img, style_img, generated, extractor, style_layers, style_weights, lbfgs_lr, lbfgs_iters, alpha, beta):
    """
    Refines a generated image using the L-BFGS optimization algorithm for neural style transfer.

    Args:
        content_img (torch.Tensor): The content image tensor.
        style_img (torch.Tensor): The style image tensor.
        generated (torch.Tensor): The image tensor to be optimized (typically initialized as a copy of the content image).
        extractor (callable): Feature extractor that returns a dict with 'content' and 'style' features.
        style_layers (list of str): Names of layers to use for style representation.
        style_weights (list of float): Weights for each style layer.
        lbfgs_lr (float): Learning rate for the L-BFGS optimizer.
        lbfgs_iters (int): Number of iterations for the L-BFGS optimizer.
        alpha (float): Weight for the content loss.
        beta (float): Weight for the style loss.

    Returns:
        torch.Tensor: The optimized image tensor after L-BFGS polishing.
    """
    with torch.no_grad():
        content_target = extractor(content_img)['content']
        style_targets = extractor(style_img)['style']

    content_loss_mod = loss_content(content_target)
    style_loss_mods = {l: loss_style(style_targets[l], w) for l, w in zip(style_layers, style_weights)}

    optimizer = LBFGS([generated], lr=lbfgs_lr, max_iter=lbfgs_iters, history_size=100, line_search_fn="strong_wolfe")
    lbfgs_prog_bar = tqdm(total=lbfgs_iters, desc='Using LBFGS')

    def closure():
        optimizer.zero_grad()
        feats = extractor(generated)
        content_loss_mod(feats['content'])
        c_loss = content_loss_mod.loss
        s_loss = 0.0
        for l in style_layers:
            style_loss_mods[l](feats['style'][l])
            s_loss += style_loss_mods[l].loss
        total = alpha * c_loss + beta * s_loss
        total.backward()
        lbfgs_prog_bar.update(1)
        lbfgs_prog_bar.set_postfix({'Loss': f'{total.item():.6f}'})
        return total

    optimizer.step(closure)
    return generated

def gram_matrix(feature_map):
    """
    Computes the normalized Gram matrix for a given feature map tensor.

    The Gram matrix is used to measure the correlations between the different feature channels.
    It is commonly used in style transfer tasks to capture style information from images.

    Args:
        feature_map (torch.Tensor): A 4D tensor of shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: The normalized Gram matrix of shape (batch_size * channels, batch_size * channels).
    """
    """Normalized Gram matrix"""
    a, b, c, d = feature_map.size()  # a=batch
    features = feature_map.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class loss_content(nn.Module):
    """
    Content loss module for neural style transfer.

    A PyTorch module that computes L1 loss between input features and target features.
    The loss is calculated during the forward pass but the input is returned unchanged,
    allowing this loss to be integrated into a computational graph.

    Attributes:
        target (torch.Tensor): The target feature tensor (detached from the computation graph).
        loss (torch.Tensor): The computed L1 loss between input and target.

    Args:
        target (torch.Tensor): The target feature representation to match during style transfer.
    """
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = l1_loss(x, self.target)
        return x


class loss_style(nn.Module):
    """
    A PyTorch module that computes style loss using Gram matrix comparison.

    This module calculates the L1 loss between the Gram matrix of input features
    and a target Gram matrix, weighted by a specified factor. It's commonly used
    in neural style transfer to measure style similarity between images.

    Attributes:
        weight (float): Scaling factor for the loss value. Default is 1.0.
        target (torch.Tensor): Detached Gram matrix of the target style image.
        loss (torch.Tensor): Computed style loss (set during forward pass).

    Args:
        target (torch.Tensor): Input tensor representing the target style image.
        weight (float, optional): Multiplier for the loss. Default is 1.0.

    Returns:
        torch.Tensor: The input tensor x (unchanged), with loss stored in self.loss.
    """
    def __init__(self, target, weight=1.0):
        super().__init__()
        self.weight = weight
        self.target = gram_matrix(target).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = l1_loss(G, self.target) * self.weight
        return x


class FeatureExtractor(nn.Module):
    """
    FeatureExtractor Module for Neural Style Transfer

    A PyTorch module that extracts feature maps from specified layers of a pre-trained VGG19 network.
    This is commonly used in neural style transfer to compute content and style losses.

    Attributes:
        vgg (nn.Module): Pre-trained VGG19 feature extractor (frozen, no gradient computation).
        layer_map (dict): Maps layer names (e.g., 'relu1_1') to their indices in the VGG19 architecture.
        content_idx (int): Index of the layer used for content feature extraction.
        style_idxs (list): List of indices for layers used in style feature extraction.
        max_idx (int): Maximum layer index to process (optimization to avoid unnecessary computation).
    """
    def __init__(self, content_layer, style_layers):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

        self.layer_map = {
            'relu1_1': 1, 'relu1_2': 4,
            'relu2_1': 6, 'relu2_2': 9,
            'relu3_1': 11, 'relu3_2': 14, 'relu3_3': 17, 'relu3_4': 20,
            'relu4_1': 22, 'relu4_2': 25, 'relu4_3': 28, 'relu4_4': 31,
            'relu5_1': 33, 'relu5_2': 36, 'relu5_3': 39, 'relu5_4': 42,
        }
        self.content_idx = self.layer_map[content_layer]
        self.style_idxs = [self.layer_map[l] for l in style_layers]
        self.max_idx = max(self.content_idx, max(self.style_idxs))

    def forward(self, x):
        content_feat = None
        style_feats = {layer: None for layer in self.style_idxs}

        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i == self.content_idx:
                content_feat = x
            if i in self.style_idxs:
                for name, idx in self.layer_map.items():
                    if idx == i:
                        style_feats[name] = x
            if i >= self.max_idx:
                break

        return {
            'content': content_feat,
            'style': {name: feat for name, feat in style_feats.items() if feat is not None}
        }

def apply_histogram_matching(generated, target):
        """
        Apply histogram matching to align the color distribution of generated image to target image.
        
        Args:
            generated: Input image (numpy array) with shape (height, width, channels) to be matched.
            target: Reference image (numpy array) with shape (height, width, channels) whose histogram will be used as reference.
        
        Returns:
            torch.Tensor: Histogram-matched image as a PyTorch tensor with shape (channels, height, width).
        """
        matched = match_histograms(generated, target, channel_axis=2)
        return torch.from_numpy(matched).permute((2, 0, 1))