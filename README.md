# Guiding Diffusion Models with Semantically Degraded Conditions

**[CVPR 2026] Official Implementation**

**English** | [中文](README_zh.md)

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

![Title Image](figures/title_image_01.png)

![Method Overview](figures/pipeline_01.png)

## Abstract

Classifier-Free Guidance (CFG) is a cornerstone of modern text-to-image models, yet its reliance on a semantically vacuous null prompt generates a guidance signal prone to geometric entanglement. This is a key factor limiting its precision, leading to well-documented failures in complex compositional tasks. We propose **Condition-Degradation Guidance (CDG)**, a novel paradigm that replaces the null prompt with a strategically degraded condition. This reframes guidance from a coarse "good vs. null" contrast to a more refined "good vs. almost good" discrimination, thereby compelling the model to capture fine-grained semantic distinctions. We find that tokens in transformer text encoders split into two functional roles: content tokens encoding object semantics, and context-aggregating tokens capturing global context. By selectively degrading only the former—a strategy we call **stratified degradation**—CDG constructs degraded conditions without external models or training. Validated across diverse architectures including Stable Diffusion 3, SD3.5, FLUX, and Qwen-Image, CDG markedly improves compositional accuracy and text-image alignment. As a lightweight, plug-and-play module, it achieves this with negligible computational overhead. Our work challenges the reliance on static, information-sparse negative samples and establishes a new principle for diffusion guidance: the construction of adaptive, semantically-aware negative samples is critical to achieving precise semantic control.

## Highlights

- We reveal a functional dichotomy in transformer text encoders between **content tokens** (encoding object-specific semantics) and **context-aggregating tokens** (encoding global compositional context), and propose **stratified degradation** as a principled strategy for constructing semantically degraded negative conditions.
- We instantiate this principle in **Condition-Degradation Guidance (CDG)**, a lightweight, training-free, plug-and-play module requiring no external models or additional training.
- We validate CDG across diverse models (SD3, SD3.5, FLUX.1-dev, Qwen-Image), providing geometric evidence for superior signal orthogonality and demonstrating consistent metric improvements with negligible overhead.

## Quick Start

Get started with CDG in 3 simple steps:

### 1. Install Dependencies

Create a conda environment and install required packages:

```bash
conda create -n cdg python=3.10
conda activate cdg

# Install system dependencies
conda install ffmpeg=6.1.2 -c conda-forge

# Install PyTorch with CUDA 12.1 support (IMPORTANT: include torchaudio)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install diffusers transformers pillow numpy pandas tqdm matplotlib \
    torchmetrics torch-fidelity hpsv2 aesthetic-predictor-v2-5 t2v-metrics accelerate
```


### 2. Download SD3 Model

Download Stable Diffusion 3 Medium from Hugging Face:
- Model: [stabilityai/stable-diffusion-3-medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)

The downloaded model should contain the following files:
- `config.json`
- `model.safetensors` or `pytorch_model.bin`
- Other configuration files

Update the path in `configs/models/models_path.json`:

```json
{
    "sd3": "path/to/your/stable-diffusion-3-medium"
}
```

> **Important**: Use the absolute path to the directory containing `config.json`.

### 3. Run CDG

Generate your first image with CDG:

```bash
CUDA_VISIBLE_DEVICES=0 python models/runner/cdg/sd3.py
```

The generated image will be saved in the `outputs/` directory.

---

**For detailed model setup (SD3.5, FLUX, Qwen-Image) and comparing with other methods (CFG, CADS, ICG, SEG, PAG, SFG, DNP), see the [Setup](#setup) section below.**

## Setup

### 1. Environment

Create a conda environment with Python 3.10 and install dependencies:

```bash
conda create -n cdg python=3.10
conda activate cdg

# Install system dependencies
conda install ffmpeg=6.1.2 -c conda-forge

# Install PyTorch with CUDA 12.1 support (IMPORTANT: include torchaudio)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install diffusers transformers pillow numpy pandas tqdm matplotlib \
    torchmetrics torch-fidelity hpsv2 aesthetic-predictor-v2-5 t2v-metrics accelerate
```

**Important Notes**:
- **System Dependencies**: ffmpeg is required for t2v-metrics (VQAScore evaluation). Install via conda as shown above.
- **PyTorch Installation**: Install torch, torchvision, and torchaudio from the same CUDA index (cu121) to avoid compatibility issues.
- **Hardware Requirements**: We recommend GPUs with at least 40GB VRAM for optimal performance.

### 2. Models and Datasets

#### Required Models

Configuration files use placeholder paths (`path/to/your/...`). Download the models you need and update `configs/models/models_path.json` accordingly.

**Base Models:**

| Model | Type | Link | Status |
|-------|------|------|--------|
| Stable Diffusion 3 Medium | Text-to-Image | [stabilityai/stable-diffusion-3-medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium) | Required |
| Stable Diffusion 3.5 Large | Text-to-Image | [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) | Required |
| FLUX.1-dev | Text-to-Image | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Required |
| Qwen-Image | Text-to-Image | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | Optional |
| SD3-ControlNet-Canny | Controllable Generation | [InstantX/SD3-Controlnet-Canny](https://huggingface.co/InstantX/SD3-Controlnet-Canny) | Optional |

**Supporting Models:**
- **BLIP2 Model**: [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) - Required for DNP method
- **Aesthetic Predictor**: `aesthetic_predictor_v2_5.pth` - Optional local weights for Aesthetic Score (auto-downloaded if not provided)

Update `configs/models/models_path.json`:

```json
{
    "sd3": "path/to/your/stable-diffusion-3-medium-diffusers",
    "sd35": "path/to/your/stable-diffusion-3.5-large",
    "flux": "path/to/your/FLUX.1-dev",
    "qwenimage": "path/to/your/Qwen-Image",
    "sd3_controlnet": "path/to/your/SD3-Controlnet-Canny"
}
```

**Important**: Configuration file paths are placeholders. After downloading models, update paths to your actual storage location. Similarly, dataset paths in `configs/evaluations/` also need to be updated to your actual paths.

### 3. Datasets

#### COCO 2017 Dataset

Download the COCO 2017 validation dataset:
- Images: [val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
- Annotations: [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

Update `configs/evaluations/coco2017.json` with your paths:

```json
{
    "image_path": "path/to/your/coco2017/val2017",
    "prompt_file_path": "path/to/your/coco2017/annotations/captions_val2017.json"
}
```

#### GenAI-Bench Dataset

GenAI-Bench prompts are included in `evaluation/Genai-Bench/Genai-Bench_prompts.json`. No additional download required. See `evaluation/Genai-Bench/README.md` for details.

## Reproducing Paper Results

### Evaluation Configuration

Before running experiments, you need to configure evaluation metrics (FID, CLIPScore, AestheticScore, VQAScore) and download the COCO 2017 dataset.

**Detailed configuration instructions:**
- [English: Evaluation Metrics Configuration Guide](configs/evaluations/README.md)
- [中文：评测指标配置指南](configs/evaluations/README_zh.md)

The configuration files include:
- `configs/evaluations/metrics.json`: Metric configurations and model paths
- `configs/evaluations/coco2017.json`: COCO dataset paths

### COCO 2017 Evaluation

Run the following experiments to reproduce quantitative results:

| Experiment | Models/Methods | Command |
|------------|----------------|---------|
| COCO-SD3 | SD3 + CFG/CADS/ICG/CDG/SEG/PAG/DNP/SFG | `./experiment/COCO-SD3/run.sh` |
| COCO-SD3.5 | SD3.5 + CFG/CADS/ICG/CDG/PAG/SEG | `./experiment/COCO-SD3.5/run.sh` |
| COCO-Flux | FLUX + CFG/CADS/ICG/CDG | `./experiment/COCO-Flux/run.sh` |

### GenAI-Bench Evaluation

| Experiment | Models/Methods | Command |
|------------|----------------|---------|
| Genai-SD3 | SD3 + CFG/CADS/ICG/CDG/SEG/PAG | `./experiment/Genai-SD3/run.sh` |
| Genai-SD3.5 | SD3.5 + CFG/CADS/ICG/CDG/SEG/PAG | `./experiment/Genai-SD3.5/run.sh` |
| Genai-Flux | FLUX + CFG/CADS/ICG/CDG | `./experiment/Genai-Flux/run.sh` |

Each experiment script will:
1. Generate images using multiple methods with distributed GPU support
2. Evaluate using FID, CLIP Score, Aesthetic Score, and VQA Score metrics
3. Save results and comparison tables in the experiment's output directory

**Note**: The scripts use `GPUS="0"` by default. Modify this in the script files to use multiple GPUs (e.g., `GPUS="0,1,2"`).

## Project Structure

```
classifier-degradation-guidance/
├── configs/               # Configuration files
│   ├── models/            # Model paths configuration
│   ├── methods/           # Method-specific parameters
│   └── evaluations/       # Dataset and metric configurations
├── models/                # Core model implementations
│   ├── pipelines/         # Custom diffusion pipelines
│   │   ├── cdg/           # CDG pipeline (ours)
│   │   ├── cads/          # CADS pipeline
│   │   ├── icg/           # ICG pipeline
│   │   └── ...
│   └── runner/            # Method runners for each model
│       ├── cdg/           # CDG runners (sd3, sd35, flux, qwenimage)
│       ├── cfg/           # CFG runners
│       └── ...
├── evaluation/            # Evaluation framework
│   ├── core/              # Core evaluation logic
│   ├── metrics/           # Metric implementations (FID, CLIP, Aesthetic, VQA)
│   └── Genai-Bench/       # GenAI-Bench dataset
├── experiment/            # Reproduction experiments
│   ├── COCO-SD3/          # SD3 on COCO
│   ├── COCO-SD3.5/        # SD3.5 on COCO
│   ├── COCO-Flux/         # FLUX on COCO
│   ├── Genai-SD3/         # SD3 on GenAI-Bench
│   ├── Genai-SD3.5/       # SD3.5 on GenAI-Bench
│   └── Genai-Flux/        # FLUX on GenAI-Bench
├── figures/               # Paper figures
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── README.md              # This file
```

## Citation

If you find our work useful, please cite our paper:

```bibtex
@inproceedings{cdg2026,
  title={Guiding Diffusion Models with Semantically Degraded Conditions},
  author={[Authors]},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## Acknowledgments

We would like to thank:
- The [Hugging Face Diffusers](https://github.com/huggingface/diffusers) team for their excellent diffusion model library
- Authors of baseline methods (CFG, CADS, ICG, SEG, PAG, SFG, DNP) for their inspiring work
- [COCO Dataset](https://cocodataset.org) and [GenAI-Bench](https://huggingface.co/datasets/TIGER-Lab/GenAI-Bench) teams for providing evaluation benchmarks

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: Third-party components (models, datasets, libraries) are subject to their respective licenses. Please review the licenses of all dependencies before use.
