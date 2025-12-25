# Neural Style Transfer (NST)

A PyTorch implementation of neural style transfer using VGG19, featuring multi-scale pyramid optimization and L-BFGS refinement for high-quality artistic image generation.

## Features

- **Multi-scale Pyramid Optimization**: Coarse-to-fine approach for better style transfer quality
- **Two-stage Optimization**: Adam optimizer for initial convergence, L-BFGS for final refinement
- **Flexible Style Control**: Configurable style layers and weights
- **Optional Histogram Matching**: Color distribution alignment with content or style image
- **GPU Acceleration**: CUDA support for faster processing
- **Visualization Tools**: Built-in utilities for comparing content, style, and generated images

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. If you don't have uv installed:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and set up the project:

```bash
git clone https://github.com/aidendorian/NeuralStlyeTransfer.git
cd NeuralStlyeTransfer
uv sync
```

## Project Structure

```
NeuralStlyeTransfer/
├── data/
│   ├── content.jpg          # Your content images
│   └── style.jpg            # Your style reference images
├── output/                  # Generated stylized images
├── src/
│   ├── __init__.py
│   ├── image_nst.py         # Main script
│   ├── utils.py             # Core NST functions
│   └── visualization.py     # Visualization utilities
├── pyproject.toml           # Project dependencies
├── README.md
└── uv.lock
```

## Usage

### Basic Usage

Place your content and style images in the `data/` directory, then run:

```bash
uv run src/image_nst.py
```

### Configuration

Edit the parameters in `src/image_nst.py`:

```python
# Style Transfer Parameters
STYLE_LAYERS = ['relu1_1', 'relu3_3', 'relu4_3']  # VGG19 layers for style
STYLE_WEIGHTS = [2.0, 1.5, 1.0]                   # Weight for each style layer
CONTENT_LAYER = 'relu4_2'                         # VGG19 layer for content

# Loss Weights
ALPHA = 1          # Content loss weight
BETA = 1e6         # Style loss weight (higher = more stylized)

# Optimization Parameters
ADAM_ITERS = 200        # Adam optimizer iterations
ADAM_LR = 0.02          # Adam learning rate
LBFGS_ITERS = 300       # L-BFGS iterations for refinement
LBFGS_LR = 1.0          # L-BFGS learning rate
PYRAMID_LEVELS = 3      # Number of pyramid scales

# Image Settings
MAX_SIDE = 1280         # Maximum image dimension (pixels)
CONTENT_PATH = 'data/content.jpg'
STYLE_PATH = 'data/style.jpg'
DEVICE = 'cuda'         # Use 'cpu' if no GPU available

# Optional: Histogram Matching
APPLY_HISTOGRAM_MATCHING = False      # Enable color distribution matching
HISTOGRAM_MATCHING_TARGET = 'content' # 'content' or 'style'
```

## Parameters Guide

### Content vs Style Balance

- **ALPHA (Content Weight)**: Controls how much the output resembles the original content
  - Higher values (e.g., 10): Preserves more content structure
  - Lower values (e.g., 0.1): Allows more stylistic freedom

- **BETA (Style Weight)**: Controls style transfer intensity
  - Common range: `1e5` to `1e7`
  - Higher values: Stronger style application
  - Lower values: Subtler style effects

### Style Layers

The model uses VGG19 ReLU layers. Common configurations:

- **Shallow layers** (`relu1_1`, `relu2_1`): Capture fine textures and patterns
- **Deep layers** (`relu4_1`, `relu5_1`): Capture abstract style features
- **Balanced** (default): Mix of shallow and deep for comprehensive style transfer

### Optimization

- **PYRAMID_LEVELS**: 
  - More levels (3-4): Better quality, longer processing
  - Fewer levels (1-2): Faster results, may lack detail

- **ADAM_ITERS** & **LBFGS_ITERS**:
  - Increase for higher quality (at cost of time)
  - Default values work well for most images

### Histogram Matching

Enable this to match color distribution:
- `'content'`: Output colors match content image
- `'style'`: Output colors match style image
- Useful when colors look washed out or oversaturated

## Output

Generated images are saved to `output/` with filenames encoding the parameters used:

```
{ALPHA}_{BETA}_{ADAM_ITERS}_{ADAM_LR}_{LBFGS_ITERS}_{LBFGS_LR}_{HISTOGRAM_MATCHING}_{TARGET}.png
```

Example: `1_1000000.0_200_0.02_300_1.0_False_content.png`

## Tips for Best Results

1. **Start with defaults**: The default parameters work well for most images
2. **Adjust BETA first**: This has the most visible impact on style intensity
3. **Match image sizes**: Similar dimensions for content and style often work better
4. **Experiment with layers**: Different style layer combinations produce varied effects
5. **GPU recommended**: Style transfer is computationally intensive; CUDA dramatically speeds up processing

## Requirements

- PyTorch with CUDA 12.8

All dependencies are managed by uv and specified in `pyproject.toml`.

## Algorithm Overview

1. **Image Pyramid**: Process images at multiple scales (coarse to fine)
2. **Adam Optimization**: Fast initial convergence at each pyramid level
3. **L-BFGS Refinement**: High-quality final polish using second-order optimization
4. **Optional Histogram Matching**: Color distribution alignment post-processing

The implementation uses VGG19 feature extraction with Gram matrices for style representation and L1 loss for both content and style matching.