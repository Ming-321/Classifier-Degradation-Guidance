# 模型实现

[English](README.md) | **中文**

本目录包含 CDG（我们的方法）及用于对比的 baseline 方法的实现。

---

## 目录结构

```
models/
├── pipelines/           # 自定义 pipeline 实现
│   ├── cdg/             # CDG（分类器退化引导）- 我们的方法
│   ├── cads/            # CADS - Baseline
│   ├── icg/             # ICG - Baseline
│   ├── seg/             # SEG - Baseline
│   ├── pag/             # PAG - Baseline
│   ├── sfg/             # SFG - Baseline
│   └── ...
└── runner/              # 各模型的方法运行器
    ├── cdg/             # CDG runners (sd3, sd35, flux, qwenimage)
    ├── cfg/             # CFG runners
    └── ...
```

---

## CDG (Classifier Degradation Guidance) - 我们的方法

- **论文**: Guiding Diffusion Models with Semantically Degraded Conditions (CVPR 2026)
- **作者**: 韩世龙, 张育铭, 王红霞
- **实现位置**: `models/pipelines/cdg/`

---

## Baseline 方法

### CFG (Classifier-Free Guidance)

- **论文**: Classifier-Free Diffusion Guidance, 2022
- **作者**: Jonathan Ho, Tim Salimans
- **实现方式**: 标准 diffusers 库

### CADS (Condition-Annealed Diffusion Sampler)

- **论文**: CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling (ICLR 2024)
- **作者**: Seyedmorteza Sadat, Jakob Buhmann, Derek Bradley, Otmar Hilliges, Romann M. Weber
- **实现方式**: 复现于 `models/pipelines/cads/`

### ICG (Independent Condition Guidance)

- **论文**: No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models (ICLR 2025)
- **作者**: Seyedmorteza Sadat, Manuel Kansy, Otmar Hilliges, Romann M. Weber
- **实现方式**: 复现于 `models/pipelines/icg/`

### SEG (Smoothed Energy Guidance)

- **论文**: Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention (NeurIPS 2024)
- **作者**: Susung Hong
- **实现方式**: 复现于 `models/pipelines/seg/`

### PAG (Perturbed Attention Guidance)

- **论文**: Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance (ECCV 2024)
- **作者**: Donghoon Ahn, Hyoungwon Cho, Jaewon Min, Wooseok Jang, Jungwoo Kim, SeonHwa Kim, Hyun Hee Park, Kyong Hwan Jin, Seungryong Kim
- **实现方式**: 复现于 `models/pipelines/pag/`

### SFG (Segmentation-Free Guidance)

- **论文**: Segmentation-Free Guidance for Text-to-Image Diffusion Models (CVPR 2024 Workshop)
- **作者**: Kian Azarian, Debasmit Das, Qiqi Hou, Fatih Porikli
- **实现方式**: 复现于 `models/pipelines/sfg/`

### DNP (Diffusion-Negative Prompting)

- **论文**: Improving Image Synthesis with Diffusion-Negative Sampling (ECCV 2024)
- **作者**: Alakh Desai, Nuno Vasconcelos
- **实现方式**: 复现于 `models/runner/dnp/`
