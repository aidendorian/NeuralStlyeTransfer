import torch
from torch.optim import LBFGS, Adam
from utils import loss_content, loss_style, FeatureExtractor
from visualization import visualize
from PIL import Image
from torchvision.transforms import Compose
from torchvision import transforms
from tqdm import tqdm

STYLE_LAYERS = [0, 5, 10, 19, 28]
CONTENT_LAYER = 21
ALPHA = 1
BETA = 700
MAX_ITER = 512
IMAGE_SIZE = 512
CONTENT_PATH = 'data/content.jpg'
STYLE_PATH = 'data/style.jpg'
ADAM_LR = 0.13
LBFGS_LR = 1.
DEVICE = 'cuda'

content_image = Image.open(CONTENT_PATH).convert('RGB')
style_image = Image.open(STYLE_PATH).convert('RGB')

image_transforms = Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

content_image = image_transforms(content_image).to(DEVICE)
style_image = image_transforms(style_image).to(DEVICE)

generated = torch.clone(content_image)
generated = generated.clamp(0, 1)
generated.requires_grad_(True)

optimizer = LBFGS([generated], lr=LBFGS_LR, max_iter=MAX_ITER)

extractor = FeatureExtractor(CONTENT_LAYER, STYLE_LAYERS).to(DEVICE)

with torch.no_grad():
    style_features_list = extractor.style_extractor(style_image)
    content_feature = extractor.content_extractor(content_image)
    style_loss_module_list = [loss_style(features) for features in style_features_list]
    content_loss_module = loss_content(content_feature)

lbfgs_prog_bar = tqdm(total=MAX_ITER, desc='Using LBFGS')

def closure():
    optimizer.zero_grad()
    generated_style_features_list = extractor.style_extractor(generated)
    generated_content_feature = extractor.content_extractor(generated)
    
    for loss_module, generated_features in zip(style_loss_module_list, generated_style_features_list):
        loss_module(generated_features)
    
    style_loss = sum([loss_module.style_loss for loss_module in style_loss_module_list])
    content_loss_module(generated_content_feature)
    content_loss = content_loss_module.content_loss
    total_loss = content_loss*ALPHA + style_loss*BETA
    # print(f'C_loss: {content_loss:.6f} | S_loss: {style_loss:.6f} | Total: {total_loss:.6}')
    total_loss.backward()
    lbfgs_prog_bar.update(1)
    lbfgs_prog_bar.set_postfix({'C_loss': f'{content_loss:.6f}', 'S_loss': f'{style_loss:.6f}', 'Total': f'{total_loss:.6f}'})
    return total_loss

optimizer.step(closure=closure)
visualize(generated, content_image, style_image, f'output/result_LBFGS_{BETA}_{MAX_ITER}_{LBFGS_LR}.png', True)

generated = torch.clone(content_image)
generated = generated.clamp(0, 1)
generated.requires_grad_(True)
optimizer = Adam([generated], lr=ADAM_LR)

with torch.no_grad():
    style_features_list = extractor.style_extractor(style_image)
    content_feature = extractor.content_extractor(content_image)
    style_loss_module_list = [loss_style(features) for features in style_features_list]
    content_loss_module = loss_content(content_feature)

adam_prog_bar = tqdm(total=MAX_ITER, desc='Using Adam')

for i in range(512):
    optimizer.zero_grad()
    generated_style_features_list = extractor.style_extractor(generated)
    generated_content_feature = extractor.content_extractor(generated)
    
    for loss_module, generated_features in zip(style_loss_module_list, generated_style_features_list):
        loss_module(generated_features)
    
    style_loss = sum([loss_module.style_loss for loss_module in style_loss_module_list])
    content_loss_module(generated_content_feature)
    content_loss = content_loss_module.content_loss
    total_loss = content_loss*ALPHA + style_loss*BETA
    # print(f'C_loss: {content_loss:.6f} | S_loss: {style_loss:.6f} | Total: {total_loss:.6}')
    adam_prog_bar.update(1)
    adam_prog_bar.set_postfix({'C_loss': f'{content_loss:.6f}', 'S_loss': f'{style_loss:.6f}', 'Total': f'{total_loss:.6f}'})
    total_loss.backward()
    optimizer.step()
    
visualize(generated, content_image, style_image, f'output/result_Adam_{BETA}_{MAX_ITER}_{ADAM_LR}.png', True)