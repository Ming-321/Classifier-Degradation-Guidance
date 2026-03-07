# SEG (Self-attention Guidance) for Flux

This directory contains the implementation of SEG (Self-attention Guidance) for Flux models.

## Overview

SEG is a guidance technique that perturbs image self-attention queries through spatial blurring. By comparing the original and spatially-degraded outputs, SEG can guide the generation towards better image quality without requiring additional training.

## Implementation

### Files

- `flux/flux_attn_processor.py`: SEG attention processor for Flux
  - `SEGFluxAttnProcessor`: Handles query perturbation via Gaussian/uniform blur
  - `gaussian_blur_2d`: 2D Gaussian blur utility function
- `flux/pipeline.py`: SEG-enabled Flux pipeline

### Key Features

- **Spatial Perturbation**: Blurs image queries to degrade self-attention
- **Blur Modes**: Gaussian blur or uniform blur (infinite sigma)
- **Three Guidance Modes**: CFG only, SEG only, CFG+SEG combined
- **Text Preservation**: Only perturbs image queries, preserves text attention
- **Layer Selection**: Apply SEG to specific transformer layers

## Usage

### Basic Usage (SEG Only)

```python
import torch
from models.pipelines.seg.flux.pipeline import SEGFluxPipeline

# Load pipeline
pipe = SEGFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Generate with SEG (SEG only mode)
image = pipe(
    "A cat holding a sign that says hello world",
    seg_scale=3.0,
    seg_blur_sigma=9999999.0,  # Uniform blur
    seg_applied_layers_index=[0],  # Apply to first block
    true_cfg_scale=1.0,  # Disable CFG
    guidance_scale=3.5,
).images[0]
```

### With Classifier-Free Guidance (CFG + SEG)

```python
image = pipe(
    "A beautiful landscape",
    negative_prompt="blurry, low quality",
    seg_scale=2.5,
    seg_blur_sigma=9999999.0,
    seg_applied_layers_index=[0, 1],
    true_cfg_scale=3.0,  # Enable CFG
    guidance_scale=3.5,
).images[0]
```

### With Gaussian Blur

```python
image = pipe(
    "A futuristic cityscape",
    seg_scale=3.0,
    seg_blur_sigma=2.0,  # Gaussian blur with sigma=2.0
    seg_applied_layers_index=[0],
).images[0]
```

## Parameters

- `seg_scale` (float, default: 3.0): SEG guidance strength. Higher values increase the effect.
- `seg_blur_sigma` (float, default: 9999999.0): Blur standard deviation.
  - Values > 9999.0: Uniform blur (all queries become mean)
  - Values 1.0-10.0: Gaussian blur with visible spatial smoothing
- `seg_applied_layers_index` (List[int], default: [0]): Transformer block indices to apply SEG.
  - Flux has `transformer_blocks` (joint attention) and `single_transformer_blocks`
  - Example: `[0, 1, 2]` applies to first three blocks

## Guidance Modes

### 1. SEG Only
- Batch: `[original, perturbed]`
- Formula: `output = original + seg_scale * (original - perturbed)`
- Use when: No negative prompt, want pure SEG guidance

### 2. CFG Only  
- Batch: `[uncond, cond]`
- Formula: `output = uncond + cfg_scale * (cond - uncond)`
- Use when: `seg_scale = 0`, standard CFG

### 3. CFG + SEG
- Batch: `[uncond, cond, cond_perturbed]`
- Formula: `output = cond + (cfg_scale - 1) * (cond - uncond) + seg_scale * (cond - cond_perturbed)`
- Use when: Want both CFG and SEG guidance combined

## How SEG Works

1. **Query Perturbation**: For each selected layer:
   - Original path: Normal query computation
   - Perturbed path: Spatially blur the image queries

2. **Spatial Blur**:
   - Reshape queries to spatial grid (H x W)
   - Apply Gaussian or uniform blur
   - Reshape back to sequence format

3. **Attention**: 
   - Compute attention with perturbed queries
   - Degraded spatial structure reduces image quality

4. **Guidance**: Compare original vs perturbed:
   ```
   output = original + seg_scale * (original - perturbed)
   ```

## Testing

Run the test script to verify the implementation:

```bash
python test_seg_flux.py
```

This will test:
- SEG only mode
- CFG + SEG mode
- Gaussian blur vs uniform blur

## References

- SEG Paper: [Self-attention Guidance](https://arxiv.org/abs/2210.00939)
- SD3 Implementation: `models/pipelines/seg/sd3/`

