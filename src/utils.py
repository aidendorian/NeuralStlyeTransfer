from torch.nn import Module
from torch import mm
from torchvision.models import vgg19, VGG19_Weights
from torch.nn.functional import l1_loss

class loss_content(Module):
    """Calculate the Content Loss

    Args:
        target_features : Content Image Feature Map
    """
    def __init__(self, target_features):
        super().__init__()
        self.target = target_features.detach()
        self.content_loss = None
        
    def forward(self, x):
        self.content_loss = l1_loss(x, self.target)
        return x

def gram_matrix(feature_map):
    """Generates Gram matrix of the feature Map
    Args:
        feature_map : Feature Map
    Returns:
        Gram Matrix
    """
    c, h, w = feature_map.shape
    features = feature_map.view(c, h*w)
    gram = mm(features, features.t())
    return gram.div(c*h*w)

class loss_style(Module):
    """Calculate the Style Loss
    Args:
        target_features : Style Image Feature Map from ith Layer of VGG19
    """
    def __init__(self, target_features):
        super().__init__()
        self.target_features = gram_matrix(target_features.detach()).detach()
        self.style_loss = 0.
        
    def forward(self, x):
        G = gram_matrix(x)
        self.style_loss = l1_loss(G, self.target_features)
        return x
    
class FeatureExtractor(Module):
    """Extracts Feature Maps from Specific Layers of VGG19
    Args:
        style_layers : List of Indexes of layers from which style features need to be extracted from.
        content_layer : Index of the layer from which the content feature need to extracted from.
    """
    def __init__(self, content_layer, style_layers, device='cuda'):
        super().__init__()
        extractor_features = vgg19(weights=VGG19_Weights.DEFAULT).to(device).features.eval()
        
        for params in extractor_features.parameters():
            params.requires_grad = False
            
        self.style_layers = style_layers
        self.content_layer = content_layer
        self.vgg19 = extractor_features
        
    def style_extractor(self, x):
        """Extracts Feature Maps from the specific VGG19 layers
        Args:
            x : Image Tensor from which Style Features need to be extacted. 
        Returns:
            features : List of all the feature maps extracted.
        """
        features = []
        
        for i, layer in enumerate(self.vgg19):
            x = layer(x)
            if i in self.style_layers:
                features.append(x)
            if i == max(self.style_layers):
                break
        return features
        
    def content_extractor(self, x):
        """Extracts Feature Maps from the specific VGG19 layers
        Args:
            x : Image Tensor from which Content Features need to be extacted. 
        Returns:
            features : Content feature map extracted.
        """
        features = 0.
        
        for i, layer in enumerate(self.vgg19):
            x = layer(x)
            if i == self.content_layer:
                features = x
                break
        return features