from torch.nn import MSELoss, Module
from torch import bmm
from torchvision.models import vgg19

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
    
class FeatureExtractor(Module):
    """Extracts Feature Maps from Specific Layers of VGG19

    Args:
        style_layers : List of Indexes of layers from which style features need to be extracted from.
        content_layer : Index of the layer from which the content feature need to extracted from.
    """
    def __init__(self, style_layers, content_layer):
        super().__init__()
        extractor_model = vgg19(weights='DEFAULT')
        extractor_features = extractor_model.features
        
        for params in extractor_features.parameters():
            params.requires_grad(False)
            
        self.style_layers = style_layers
        self.content_layer = content_layer
        self.vgg19 = extractor_features
        
    def style_extractor(self, x):
        """Extracts Feature Maps from the specific VGG19 layers

        Args:
            x : Image Tensor from which Style Features need to be extacted. 

        Returns:
            features : Dictionary of all the feature maps extracted.
        """
        features = {}
        
        for i, layer in enumerate(self.vgg19):
            x = layer(x)
            
            if i in self.style_layers:
                features[f'style{i}'] = x
                
            if i == max(self.style_layers):
                break
                
        return features
        
    def content_extractor(self, x):
        """Extracts Feature Maps from the specific VGG19 layers

        Args:
            x : Image Tensor from which Content Features need to be extacted. 

        Returns:
            features : Dictionary of all the feature maps extracted.
        """
        features = {}
        
        for i, layer in enumerate(self.vgg19):
            x = layer(x)
            
            if i == self.content_layer:
                features['content21'] = x
                break
                
        return features
        
    def forward(self, x):
        features = {}
        
        for i, layer in enumerate(self.vgg19):
            x = layer(x)
            
            if i in self.style_layers:
                features[f'style{i}_1'] = x
                
            if i == self.content_layer:
                features['content21'] = x
                
            if i == max(self.style_layers):
                break
        return features