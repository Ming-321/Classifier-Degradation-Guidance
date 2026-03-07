# Evaluation Metrics Configuration Guide

**English** | [中文](README_zh.md)

This directory contains configuration files for evaluation metrics. This project uses four main evaluation metrics: **FID**, **CLIPScore**, **AestheticScore**, and **VQAScore**.

## 📋 Metrics Overview

| Metric | Purpose | Model Size | Configuration Method |
|--------|---------|------------|----------------------|
| **FID** | Image quality & distribution consistency | Small (auto-download) | No configuration needed |
| **CLIPScore** | Text-image alignment | ~600MB | Config file only |
| **AestheticScore** | Image aesthetic quality | ~1.7GB + weights | Config file only |
| **VQAScore** | Visual question answering score | ~24GB + Vision Tower | Config file + source code modification |

---

## 🚀 Quick Start

### 1. FID (Fréchet Inception Distance)

FID uses the InceptionV3 model. The model is small and will be automatically downloaded to PyTorch cache directory (usually `~/.cache/torch/hub/`) on first run, no additional configuration needed.

**Configuration** (`metrics.json`):
```json
"FID": {
    "feature_dim": 2048,
    "batch_size": 250,
    "device": "cuda"
}
```

### 2. CLIPScore

**Required Models**:
- `openai/clip-vit-base-patch32` (~600MB)

**Steps**:

#### 2.1 Download Model

Download from HuggingFace:
- Model link: https://huggingface.co/openai/clip-vit-base-patch32
- Model ID: `openai/clip-vit-base-patch32`

Download to any local directory (can use `huggingface-cli`, `git-lfs`, or other download tools).

#### 2.2 Modify Configuration File

Edit `configs/evaluations/metrics.json`:

```json
"CLIPScore": {
    "name": "path/to/your/clip-vit-base-patch32",
    "batch_size": 250,
    "device": "cuda"
}
```

Change the `name` field to your downloaded model's local path.

### 3. AestheticScore

**Required Dependencies**:
- Python package: `aesthetic-predictor-v2-5` (install from PyPI)
- SigLIP model: `google/siglip-so400m-patch14-384` (~1.7GB)
- Aesthetic Predictor weights: `aesthetic_predictor_v2_5.pth` (~2.6MB, optional, pip package can auto-download)

**Steps**:

#### 3.1 Install Dependencies

AestheticScore depends on `aesthetic-predictor-v2-5` library, please install first (can install via pip):
- PyPI package name: `aesthetic-predictor-v2-5`
- GitHub repository: https://github.com/discus0434/aesthetic-predictor-v2-5

#### 3.2 Download SigLIP Encoder

Download from HuggingFace:
- Model link: https://huggingface.co/google/siglip-so400m-patch14-384
- Model ID: `google/siglip-so400m-patch14-384`

#### 3.3 (Optional) Download Aesthetic Predictor Weights

Weight file can be obtained from:
- GitHub repository: `discus0434/aesthetic-predictor-v2-5` (check models folder)
- Filename: `aesthetic_predictor_v2_5.pth`

**Note**: If you don't manually specify the weight path, the library will auto-download from GitHub.

#### 3.4 Modify Configuration File

Edit `configs/evaluations/metrics.json`:

```json
"AestheticScore": {
    "encoder_model": "path/to/your/siglip-so400m-patch14-384",
    "predictor_path": "path/to/your/aesthetic_predictor_v2_5.pth",
    "batch_size": 250,
    "device": "cuda"
}
```

If you don't specify `predictor_path`, it will auto-download from GitHub:

```json
"AestheticScore": {
    "encoder_model": "path/to/your/siglip-so400m-patch14-384",
    "batch_size": 250,
    "device": "cuda"
}
```

### 4. VQAScore

**Required Models**:
- `clip-flant5-xxl` (~24GB)
- `clip-vit-large-patch14-336` (~1.7GB) - Vision Tower

**⚠️ Note**: VQAScore requires modifying `t2v_metrics` library source code to use local models.

**Steps**:

#### 4.1 Install Dependencies

VQAScore depends on `t2v_metrics` library, please install first (can install via pip):
- PyPI package name: `t2v_metrics`
- GitHub repository: https://github.com/linzhiqiu/t2v_metrics

#### 4.2 Download Models

Download two models from HuggingFace:

**Main Model**:
- Model link: https://huggingface.co/zhiqiulin/clip-flant5-xxl
- Model ID: `zhiqiulin/clip-flant5-xxl` (~24GB)

**Vision Tower**:
- Model link: https://huggingface.co/openai/clip-vit-large-patch14-336
- Model ID: `openai/clip-vit-large-patch14-336` (~1.7GB)

#### 4.3 Modify t2v_metrics Source Code

First find the installation location of `t2v_metrics` (can check via `python -c "import t2v_metrics; print(t2v_metrics.__file__)"`, usually in Python environment's `site-packages/t2v_metrics/` directory).

**File to modify**:  
`{site-packages}/t2v_metrics/models/vqascore_models/clip_t5_model.py`

**Modifications**:

Find these two lines (approximately in the middle of the file):

```python
'path': 'google/flan-t5-xxl'
'path': 'zhiqiulin/clip-flant5-xxl'
```

Change them to your local paths:

```python
'path': 'path/to/your/clip-flant5-xxl'
'path': 'path/to/your/clip-flant5-xxl'  # Both need to be changed
```

#### 4.4 Modify Model config.json

Edit your downloaded model configuration file:  
`path/to/your/clip-flant5-xxl/config.json`

Find the `mm_vision_tower` field and change it to local Vision Tower path:

```json
{
  "mm_vision_tower": "path/to/your/clip-vit-large-patch14-336",
  ...
}
```

#### 4.5 Modify Configuration File

Edit `configs/evaluations/metrics.json`:

```json
"VQAScore": {
    "name": "clip-flant5-xxl",
    "question_template": "Does this figure show \"{}\"? Please answer yes or no.",
    "answer_template": "Yes",
    "batch_size": 250,
    "device": "cuda"
}
```

**Note**: Keep the `name` field as `"clip-flant5-xxl"` because the actual path has been configured in the source code.

#### 4.6 (Optional) Fix transformers Compatibility Issues

If you encounter errors like `ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'`, you need to modify these files:

- `{site-packages}/t2v_metrics/models/vqascore_models/lavis/models/med.py`
- `{site-packages}/t2v_metrics/models/vqascore_models/lavis/models/blip_models/nlvr_encoder.py`
- `{site-packages}/t2v_metrics/models/vqascore_models/lavis/models/blip2_models/Qformer.py`

Change:

```python
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
```

To:

```python
from transformers.modeling_utils import PreTrainedModel
try:
    from transformers.modeling_utils import (
        apply_chunking_to_forward,
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )
except ImportError:
    from transformers.pytorch_utils import (
        apply_chunking_to_forward,
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )
```

#### 4.7 Troubleshooting: Missing Dependencies

**Problem 1: ModuleNotFoundError: No module named 'llava', 'flash_attn', or 'pytorchvideo'**

t2v_metrics 3.0 includes several optional model modules (llavaov, llavavideo, internvideo2, languagebind) that require additional dependencies:
- `llava` module (requires `llava-torch`, which conflicts with PyTorch 2.5+)
- `flash_attn` module (requires flash-attention, which needs compilation)
- `languagebind` module (requires `pytorchvideo`, which has additional dependencies)

**Solution**: Modify t2v_metrics source code to make these imports optional:

1. Edit `{site-packages}/t2v_metrics/models/vqascore_models/__init__.py`:
   ```python
   # Change line 6 from:
   from .llavaov_model import LLAVA_OV_MODELS, LLaVAOneVisionModel
   # To:
   try:
       from .llavaov_model import LLAVA_OV_MODELS, LLaVAOneVisionModel
   except ImportError:
       LLAVA_OV_MODELS = []
       LLaVAOneVisionModel = None
   
   # Change line 19 from:
   from .llavavideo_model import LLAVA_VIDEO_MODELS, LLaVAVideoModel
   # To:
   try:
       from .llavavideo_model import LLAVA_VIDEO_MODELS, LLaVAVideoModel
   except ImportError:
       LLAVA_VIDEO_MODELS = []
       LLaVAVideoModel = None
   ```

2. Edit `{site-packages}/t2v_metrics/models/clipscore_models/__init__.py`:
   ```python
   # Change line 6 from:
   from .internvideo2_clip_model import INTERNVIDEO2_CLIP_MODELS, InternVideo2CLIPScoreModel
   # To:
   try:
       from .internvideo2_clip_model import INTERNVIDEO2_CLIP_MODELS, InternVideo2CLIPScoreModel
   except ImportError:
       INTERNVIDEO2_CLIP_MODELS = []
       InternVideo2CLIPScoreModel = None
   ```

3. Edit `{site-packages}/t2v_metrics/models/itmscore_models/__init__.py`:
   ```python
   # Change line 3 from:
   from .internvideo2_itm_model import INTERNVIDEO2_ITM_MODELS, InternVideo2ITMScoreModel
   # To:
   try:
       from .internvideo2_itm_model import INTERNVIDEO2_ITM_MODELS, InternVideo2ITMScoreModel
   except ImportError:
       INTERNVIDEO2_ITM_MODELS = []
       InternVideo2ITMScoreModel = None
   ```

4. Edit `{site-packages}/t2v_metrics/models/clipscore_models/__init__.py` (additional fix):
   ```python
   # Also add after the internvideo2 fix:
   try:
       from .languagebind_video_clip_model import LANGUAGEBIND_VIDEO_CLIP_MODELS, LanguageBindVideoCLIPScoreModel
   except ImportError:
       LANGUAGEBIND_VIDEO_CLIP_MODELS = []
       LanguageBindVideoCLIPScoreModel = None
   ```

**Note**: These modifications are safe because we only use `clip-flant5-xxl` model, which doesn't depend on llava, flash_attn, or languagebind.

**Problem 2: RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions**

This occurs when torchaudio is installed from a different source than torch.

**Solution**: Reinstall torchaudio from the same PyTorch index:
```bash
pip uninstall -y torchaudio
pip install torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 📝 Complete Configuration Example

Here's a complete example of `metrics.json` (using local models):

```json
{
    "CLIPScore": {
        "name": "path/to/your/clip-vit-base-patch32",
        "batch_size": 250,
        "device": "cuda"
    },
    "VQAScore": {
        "name": "clip-flant5-xxl",
        "question_template": "Does this figure show \"{}\"? Please answer yes or no.",
        "answer_template": "Yes",
        "batch_size": 250,
        "device": "cuda"
    },
    "FID": {
        "feature_dim": 2048,
        "batch_size": 250,
        "device": "cuda"
    },
    "AestheticScore": {
        "encoder_model": "path/to/your/siglip-so400m-patch14-384",
        "predictor_path": "path/to/your/aesthetic_predictor_v2_5.pth",
        "batch_size": 250,
        "device": "cuda"
    }
}
```

---

## 🔧 FAQ

### Q1: Why not provide automated download scripts?

These model files are large (total ~28GB), downloading may take a long time and may be subject to network restrictions. Users can choose appropriate download methods (direct connection, proxy, mirror sites, etc.) based on their network environment.

### Q2: Can I use HuggingFace IDs instead of local paths?

- **CLIPScore** and **AestheticScore**: Yes, change the path to HuggingFace ID (like `"openai/clip-vit-base-patch32"`), the model will auto-download to cache directory
- **VQAScore**: Not recommended, as the `t2v_metrics` library will attempt multiple downloads, which is slow and may fail

### Q3: How to verify if configuration is correct?

When running evaluation, the code will output the loaded model paths. If you see output similar to the following, the configuration is correct:

```
Loading CLIP model: path/to/your/clip-vit-base-patch32...
CLIP model loaded successfully
```

### Q4: What happens if I update t2v_metrics after modifying source code?

If you update the library via `pip install --upgrade t2v_metrics`, your modifications will be overwritten and need to be reapplied. It's recommended to record your modifications or backup the modified files.

---

## 📚 References

- [CLIP (HuggingFace)](https://huggingface.co/openai/clip-vit-base-patch32)
- [SigLIP (HuggingFace)](https://huggingface.co/google/siglip-so400m-patch14-384)
- [Aesthetic Predictor v2.5 (GitHub)](https://github.com/discus0434/aesthetic-predictor-v2-5)
- [Aesthetic Predictor v2.5 (PyPI)](https://pypi.org/project/aesthetic-predictor-v2-5/)
- [t2v_metrics (GitHub)](https://github.com/linzhiqiu/t2v_metrics)
- [clip-flant5-xxl (HuggingFace)](https://huggingface.co/zhiqiulin/clip-flant5-xxl)
- [clip-vit-large-patch14-336 (HuggingFace)](https://huggingface.co/openai/clip-vit-large-patch14-336)
