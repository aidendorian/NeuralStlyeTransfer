from torch.nn import MSELoss, Module
from torch import bmm

def content_loss(content_fm, generated_content_fm):
    """Calculates Content Loss

    Args:
        content : Feature Map of the Content Image from the conv4_2 layer of VGG19
        generated_content : Feature Map of the Generated Image from the conv4_2 layer of VGG19
    
    Returns:
        Content Loss
    """
    loss_fn = MSELoss()
    content_loss = loss_fn(generated_content_fm, content_fm)
    return content_loss

def gram_matrix(feature_map):
    """Generates Gram matrix of the feature Map

    Args:
        feature_map : Feature Map

    Returns:
        Gram Matrix
    """
    b, c, h, w = feature_map.shape
    features = feature_map.view(b, c, h*w)
    gram = bmm(features, features.transpose(1, 2))/(b*c*h*w)
    return gram

class StyleLoss(Module):
    """Calculate the Style Loss

    Args:
        target_features : Style Image Feature Map of ith Layer of VGG19
    """
    def __init__(self, target_features):
        super().__init__()
        self.target_features = gram_matrix(target_features).detach()
        self.style_loss = 0.
        
    def forward(self, x):
        G = gram_matrix(x)
        loss_fn = MSELoss()
        self.style_loss = loss_fn(G, self.target_features)
        return x