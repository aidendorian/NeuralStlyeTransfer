from utils import load_image, run_pyramid, lbfgs_polish, FeatureExtractor, apply_histogram_matching
from visualization import visualize

STYLE_LAYERS = ['relu1_1', 'relu3_3', 'relu4_3']
STYLE_WEIGHTS = [2.0, 1.5, 1.0]
CONTENT_LAYER = 'relu4_2'
ALPHA = 1e2
BETA = 1e6
MAX_SIDE = 1280          
CONTENT_PATH = 'data/content.jpg'
STYLE_PATH = 'data/style.jpg'
DEVICE = 'cuda'
ADAM_ITERS = 200
LBFGS_ITERS = 300
PYRAMID_LEVELS = 3
ADAM_LR = 0.02
LBFGS_LR = 1.0
APPLY_HISTOGRAM_MATCHING = False
HISTOGRAM_MATCHING_TARGET = 'content' # or style


content_img, content_size = load_image(CONTENT_PATH)
style_img, _ = load_image(STYLE_PATH)

content_img, style_img = content_img.to(DEVICE), style_img.to(DEVICE)

extractor = FeatureExtractor(CONTENT_LAYER, STYLE_LAYERS).to(DEVICE)

generated = run_pyramid(content_img=content_img,
                        style_img=style_img,
                        style_layers=STYLE_LAYERS,
                        style_weights=STYLE_WEIGHTS,
                        extractor=extractor,
                        pyramid_levels=PYRAMID_LEVELS,
                        iters=ADAM_ITERS,
                        alpha=ALPHA,
                        beta=BETA)

generated = lbfgs_polish(content_img=content_img,
                         style_img=style_img,
                         generated=generated,
                         extractor=extractor,
                         style_layers=STYLE_LAYERS,
                         style_weights=STYLE_WEIGHTS,
                         lbfgs_lr=LBFGS_LR,
                         lbfgs_iters=LBFGS_ITERS,
                         alpha=ALPHA,
                         beta=BETA)

if APPLY_HISTOGRAM_MATCHING:
    if HISTOGRAM_MATCHING_TARGET == 'content':
        target = content_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    else:
        target = style_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    generated = generated.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    generated = apply_histogram_matching(generated, target)
    generated = generated.unsqueeze(0).to(DEVICE)
    
visualize(generated, content_img, style_img,
          f'output/{ALPHA}_{BETA}_{ADAM_ITERS}_{ADAM_LR}_{LBFGS_ITERS}_{LBFGS_LR}_{APPLY_HISTOGRAM_MATCHING}_{HISTOGRAM_MATCHING_TARGET}.png',
          normalized=True)