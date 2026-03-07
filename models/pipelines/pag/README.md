# PAG (Perturbed Attention Guidance) for Flux

This directory contains the implementation of PAG (Perturbed Attention Guidance) for Flux models.

## Overview

PAG is a guidance technique that perturbs self-attention between image patches by applying an identity mask. This forces the model to rely more on text conditioning and cross-attention, resulting in improved image quality and text-image alignment.

## Implementation

### Files

- `pag_utils.py`: Core PAG mixin class and utility functions
- `flux/pag_attn_processor.py`: PAG attention processors for Flux
  - `PAGFluxAttnProcessor`: PAG without CFG
  - `PAGCFGFluxAttnProcessor`: PAG with CFG
- `flux/pipeline.py`: PAG-enabled Flux pipeline

### Key Features

- **Identity Mask**: Blocks self-attention between image patches (except diagonal)
- **Text Preservation**: Only perturbs image attention, preserves text attention
- **CFG Compatible**: Works with and without classifier-free guidance
- **Layer Selection**: Apply PAG to specific transformer layers

## Usage

### Basic Usage (PAG Only)

```python
import torch
from models.pipelines.pag.flux.pipeline import PAGFluxPipeline

# Load pipeline with PAG
pipe = PAGFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    pag_applied_layers=["transformer_blocks.10"]  # Apply to block 10
)
pipe.to("cuda")

# Generate with PAG
image = pipe(
    "A cat holding a sign that says hello world",
    pag_scale=3.0,
    guidance_scale=3.5,
    num_inference_steps=28,
).images[0]
```

### With Classifier-Free Guidance

```python
image = pipe(
    "A beautiful landscape",
    negative_prompt="blurry, low quality",
    pag_scale=2.0,
    true_cfg_scale=3.0,  # Enable CFG
    guidance_scale=3.5,
).images[0]
```

### Parameters

- `pag_scale` (float, default: 3.0): PAG guidance strength. Higher values increase the effect.
- `pag_applied_layers` (str or List[str]): Transformer layers to apply PAG.
  - Examples: `"transformer_blocks.10"`, `["transformer_blocks.8", "transformer_blocks.10"]`
- `pag_adaptive_scale` (float, default: 0.0): Adaptive scaling factor that varies with timestep.

## How PAG Works

1. **Perturbation**: For each selected layer, PAG creates two attention paths:
   - Original path: Normal attention computation
   - Perturbed path: Identity mask blocks image-to-image attention

2. **Identity Mask**: 
   - Sets attention weights between image patches to -inf (except diagonal)
   - Preserves image-to-text cross-attention
   - Forces reliance on text conditioning

3. **Guidance**: Combines original and perturbed predictions:
   ```
   output = original + pag_scale * (original - perturbed)
   ```

## Testing

Run the test script to verify the implementation:

```bash
python test_pag_flux.py
```

## References

- PAG Paper: [Perturbed Attention Guidance](https://arxiv.org/abs/2403.17377)
- Original Implementation: [diffusers PAG](https://github.com/huggingface/diffusers)

