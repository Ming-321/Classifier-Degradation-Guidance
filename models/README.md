# Models Implementation

**English** | [中文](README_zh.md)

This directory contains the implementation of CDG (our method) and baseline methods for comparison.

---

## Directory Structure

```
models/
├── pipelines/           # Custom pipeline implementations
│   ├── cdg/             # CDG (Classifier Degradation Guidance) - Our method
│   ├── cads/            # CADS - Baseline
│   ├── icg/             # ICG - Baseline
│   ├── seg/             # SEG - Baseline
│   ├── pag/             # PAG - Baseline
│   ├── sfg/             # SFG - Baseline
│   └── ...
└── runner/              # Method runners for different models
    ├── cdg/             # CDG runners (sd3, sd35, flux, qwenimage)
    ├── cfg/             # CFG runners
    └── ...
```

---

## CDG (Classifier Degradation Guidance) - Our Method

- **Paper**: Guiding Diffusion Models with Semantically Degraded Conditions (CVPR 2026)
- **Authors**: Shilong Han, Yuming Zhang, Hongxia Wang
- **Implementation**: `models/pipelines/cdg/`

---

## Baseline Methods

### CFG (Classifier-Free Guidance)

- **Paper**: Classifier-Free Diffusion Guidance, 2022
- **Authors**: Jonathan Ho, Tim Salimans
- **Implementation**: Standard diffusers library

### CADS (Condition-Annealed Diffusion Sampler)

- **Paper**: CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling (ICLR 2024)
- **Authors**: Seyedmorteza Sadat, Jakob Buhmann, Derek Bradley, Otmar Hilliges, Romann M. Weber
- **Implementation**: Reproduced in `models/pipelines/cads/`

### ICG (Independent Condition Guidance)

- **Paper**: No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models (ICLR 2025)
- **Authors**: Seyedmorteza Sadat, Manuel Kansy, Otmar Hilliges, Romann M. Weber
- **Implementation**: Reproduced in `models/pipelines/icg/`

### SEG (Smoothed Energy Guidance)

- **Paper**: Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention (NeurIPS 2024)
- **Authors**: Susung Hong
- **Implementation**: Reproduced in `models/pipelines/seg/`

### PAG (Perturbed Attention Guidance)

- **Paper**: Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance (ECCV 2024)
- **Authors**: Donghoon Ahn, Hyoungwon Cho, Jaewon Min, Wooseok Jang, Jungwoo Kim, SeonHwa Kim, Hyun Hee Park, Kyong Hwan Jin, Seungryong Kim
- **Implementation**: Reproduced in `models/pipelines/pag/`

### SFG (Segmentation-Free Guidance)

- **Paper**: Segmentation-Free Guidance for Text-to-Image Diffusion Models (CVPR 2024 Workshop)
- **Authors**: Kian Azarian, Debasmit Das, Qiqi Hou, Fatih Porikli
- **Implementation**: Reproduced in `models/pipelines/sfg/`

### DNP (Diffusion-Negative Prompting)

- **Paper**: Improving Image Synthesis with Diffusion-Negative Sampling (ECCV 2024)
- **Authors**: Alakh Desai, Nuno Vasconcelos
- **Implementation**: Reproduced in `models/runner/dnp/`
