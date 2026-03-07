# 评测指标配置指南

[English](README.md) | **中文**

本目录包含评测指标的配置文件。本项目使用四个主要评测指标：**FID**、**CLIPScore**、**AestheticScore** 和 **VQAScore**。

## 💡 下载工具推荐

**国内用户推荐**：如果从 HuggingFace 下载模型速度较慢，可以使用 `hfd.sh` 下载脚本（支持断点续传、多线程下载、镜像加速）：

- [hfd.sh 官方文档](https://hf-mirror.com/)
- [hfd.sh Gist (作者: padeoe)](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f)
- [HuggingFace 镜像使用指南（知乎）](https://zhuanlan.zhihu.com/p/663712983)

## 📋 评测指标概述

| 指标 | 用途 | 模型大小 | 配置方式 |
|------|------|----------|----------|
| **FID** | 图像质量与分布一致性 | 小（自动下载） | 无需配置 |
| **CLIPScore** | 图文对齐度 | ~600MB | 配置文件 |
| **AestheticScore** | 图像美学质量 | ~1.7GB + 权重文件 | 配置文件 |
| **VQAScore** | 视觉问答评分 | ~24GB + Vision Tower | 配置文件 + 源码修改 |

---

## 🚀 快速开始

### 1. FID（Fréchet Inception Distance）

FID 使用 InceptionV3 模型，模型较小，会在首次运行时自动下载到 PyTorch 缓存目录（通常是 `~/.cache/torch/hub/`），无需额外配置。

**配置项**（`metrics.json`）：
```json
"FID": {
    "feature_dim": 2048,
    "batch_size": 250,
    "device": "cuda"
}
```

### 2. CLIPScore

**需要的模型**：
- `openai/clip-vit-base-patch32` (~600MB)

**步骤**：

#### 2.1 下载模型

从 HuggingFace 下载模型：
- 模型链接：https://huggingface.co/openai/clip-vit-base-patch32
- 模型 ID：`openai/clip-vit-base-patch32`

下载到本地任意目录即可（可使用 `huggingface-cli`、`git-lfs` 或其他下载工具）。

#### 2.2 修改配置文件

编辑 `configs/evaluations/metrics.json`：

```json
"CLIPScore": {
    "name": "path/to/your/clip-vit-base-patch32",
    "batch_size": 250,
    "device": "cuda"
}
```

将 `name` 字段改为你下载的模型本地路径即可。

### 3. AestheticScore

**需要的依赖**：
- Python 包：`aesthetic-predictor-v2-5` （从 PyPI 安装）
- SigLIP 模型：`google/siglip-so400m-patch14-384` (~1.7GB)
- Aesthetic Predictor 权重文件：`aesthetic_predictor_v2_5.pth` (~2.6MB，可选，pip 包可自动下载)

**步骤**：

#### 3.1 安装依赖

AestheticScore 依赖 `aesthetic-predictor-v2-5` 库，请先安装（可通过 pip 安装）：
- PyPI 包名：`aesthetic-predictor-v2-5`
- GitHub 仓库：https://github.com/discus0434/aesthetic-predictor-v2-5

#### 3.2 下载 SigLIP 编码器

从 HuggingFace 下载模型：
- 模型链接：https://huggingface.co/google/siglip-so400m-patch14-384
- 模型 ID：`google/siglip-so400m-patch14-384`

#### 3.3 （可选）下载 Aesthetic Predictor 权重

权重文件可以从以下位置获取：
- GitHub 仓库：`discus0434/aesthetic-predictor-v2-5`（查看 models 文件夹）
- 文件名：`aesthetic_predictor_v2_5.pth`

**注意**：如果不手动指定权重路径，库会自动从 GitHub 下载。

#### 3.4 修改配置文件

编辑 `configs/evaluations/metrics.json`：

```json
"AestheticScore": {
    "encoder_model": "path/to/your/siglip-so400m-patch14-384",
    "predictor_path": "path/to/your/aesthetic_predictor_v2_5.pth",
    "batch_size": 250,
    "device": "cuda"
}
```

如果不指定 `predictor_path`，则会自动从 GitHub 下载：

```json
"AestheticScore": {
    "encoder_model": "path/to/your/siglip-so400m-patch14-384",
    "batch_size": 250,
    "device": "cuda"
}
```

### 4. VQAScore

**需要的模型**：
- `clip-flant5-xxl` (~24GB)
- `clip-vit-large-patch14-336` (~1.7GB) - Vision Tower

**⚠️ 注意**：VQAScore 需要修改 `t2v_metrics` 库的源码才能使用本地模型。

**步骤**：

#### 4.1 安装依赖

VQAScore 依赖 `t2v_metrics` 库，请先安装（可通过 pip 安装）：
- PyPI 包名：`t2v_metrics`
- GitHub 仓库：https://github.com/linzhiqiu/t2v_metrics

#### 4.2 下载模型

从 HuggingFace 下载两个模型：

**主模型**：
- 模型链接：https://huggingface.co/zhiqiulin/clip-flant5-xxl
- 模型 ID：`zhiqiulin/clip-flant5-xxl` (~24GB)

**Vision Tower**：
- 模型链接：https://huggingface.co/openai/clip-vit-large-patch14-336
- 模型 ID：`openai/clip-vit-large-patch14-336` (~1.7GB)

#### 4.3 修改 t2v_metrics 源码

首先找到 `t2v_metrics` 的安装位置（可通过 `python -c "import t2v_metrics; print(t2v_metrics.__file__)"` 查看，通常在 Python 环境的 `site-packages/t2v_metrics/` 目录下）。

**需要修改的文件**：  
`{site-packages}/t2v_metrics/models/vqascore_models/clip_t5_model.py`

**修改内容**：

找到以下两行（大约在文件中间）：

```python
'path': 'google/flan-t5-xxl'
'path': 'zhiqiulin/clip-flant5-xxl'
```

将它们改为你的本地路径：

```python
'path': 'path/to/your/clip-flant5-xxl'
'path': 'path/to/your/clip-flant5-xxl'  # 两处都要改
```

#### 4.4 修改模型 config.json

编辑你下载的模型配置文件：  
`path/to/your/clip-flant5-xxl/config.json`

找到 `mm_vision_tower` 字段，改为本地 Vision Tower 路径：

```json
{
  "mm_vision_tower": "path/to/your/clip-vit-large-patch14-336",
  ...
}
```

#### 4.5 修改配置文件

编辑 `configs/evaluations/metrics.json`：

```json
"VQAScore": {
    "name": "clip-flant5-xxl",
    "question_template": "Does this figure show \"{}\"? Please answer yes or no.",
    "answer_template": "Yes",
    "batch_size": 250,
    "device": "cuda"
}
```

**注意**：`name` 字段保持为 `"clip-flant5-xxl"`，因为实际路径已经在源码中配置了。

#### 4.6 （可选）修复 transformers 兼容性问题

如果遇到类似 `ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'` 的错误，需要修改以下文件：

- `{site-packages}/t2v_metrics/models/vqascore_models/lavis/models/med.py`
- `{site-packages}/t2v_metrics/models/vqascore_models/lavis/models/blip_models/nlvr_encoder.py`
- `{site-packages}/t2v_metrics/models/vqascore_models/lavis/models/blip2_models/Qformer.py`

将：

```python
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
```

改为：

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

#### 4.7 故障排除：缺少依赖

**问题 1：ModuleNotFoundError: No module named 'llava'、'flash_attn' 或 'pytorchvideo'**

t2v_metrics 3.0 包含多个可选的模型模块（llavaov、llavavideo、internvideo2、languagebind），这些模块需要额外的依赖：
- `llava` 模块（需要 `llava-torch`，但其与 PyTorch 2.5+ 冲突）
- `flash_attn` 模块（需要 flash-attention，需要编译）
- `languagebind` 模块（需要 `pytorchvideo`，有额外的依赖）

**解决方案**：修改 t2v_metrics 源码，使这些导入变为可选：

1. 编辑 `{site-packages}/t2v_metrics/models/vqascore_models/__init__.py`：
   ```python
   # 将第 6 行从：
   from .llavaov_model import LLAVA_OV_MODELS, LLaVAOneVisionModel
   # 改为：
   try:
       from .llavaov_model import LLAVA_OV_MODELS, LLaVAOneVisionModel
   except ImportError:
       LLAVA_OV_MODELS = []
       LLaVAOneVisionModel = None
   
   # 将第 19 行从：
   from .llavavideo_model import LLAVA_VIDEO_MODELS, LLaVAVideoModel
   # 改为：
   try:
       from .llavavideo_model import LLAVA_VIDEO_MODELS, LLaVAVideoModel
   except ImportError:
       LLAVA_VIDEO_MODELS = []
       LLaVAVideoModel = None
   ```

2. 编辑 `{site-packages}/t2v_metrics/models/clipscore_models/__init__.py`：
   ```python
   # 将第 6 行从：
   from .internvideo2_clip_model import INTERNVIDEO2_CLIP_MODELS, InternVideo2CLIPScoreModel
   # 改为：
   try:
       from .internvideo2_clip_model import INTERNVIDEO2_CLIP_MODELS, InternVideo2CLIPScoreModel
   except ImportError:
       INTERNVIDEO2_CLIP_MODELS = []
       InternVideo2CLIPScoreModel = None
   ```

3. 编辑 `{site-packages}/t2v_metrics/models/itmscore_models/__init__.py`：
   ```python
   # 将第 3 行从：
   from .internvideo2_itm_model import INTERNVIDEO2_ITM_MODELS, InternVideo2ITMScoreModel
   # 改为：
   try:
       from .internvideo2_itm_model import INTERNVIDEO2_ITM_MODELS, InternVideo2ITMScoreModel
   except ImportError:
       INTERNVIDEO2_ITM_MODELS = []
       InternVideo2ITMScoreModel = None
   ```

4. 编辑 `{site-packages}/t2v_metrics/models/clipscore_models/__init__.py`（额外修复）：
   ```python
   # 在 internvideo2 修复之后添加：
   try:
       from .languagebind_video_clip_model import LANGUAGEBIND_VIDEO_CLIP_MODELS, LanguageBindVideoCLIPScoreModel
   except ImportError:
       LANGUAGEBIND_VIDEO_CLIP_MODELS = []
       LanguageBindVideoCLIPScoreModel = None
   ```

**注意**：这些修改是安全的，因为我们只使用 `clip-flant5-xxl` 模型，该模型不依赖 llava、flash_attn 或 languagebind。

**问题 2：RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions**

当 torchaudio 从与 torch 不同的源安装时会出现此错误。

**解决方案**：从相同的 PyTorch 索引重新安装 torchaudio：
```bash
pip uninstall -y torchaudio
pip install torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 📝 完整配置示例

以下是 `metrics.json` 的完整示例（使用本地模型）：

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

## 🔧 常见问题

### Q1: 为什么不提供自动下载脚本？

这些模型文件较大（总计 ~28GB），下载可能需要较长时间且可能受网络限制。用户可以根据自己的网络环境选择合适的下载方式（直连、代理、镜像站等）。

### Q2: 可以使用 HuggingFace ID 而不是本地路径吗？

- **CLIPScore** 和 **AestheticScore**：可以，将路径改为 HuggingFace ID（如 `"openai/clip-vit-base-patch32"`），模型会自动下载到缓存目录
- **VQAScore**：不建议，因为 `t2v_metrics` 库会尝试多次下载，速度较慢且可能失败

### Q3: 如何验证配置是否正确？

运行评测时，代码会输出加载的模型路径。如果看到类似以下输出，说明配置正确：

```
Loading CLIP model: path/to/your/clip-vit-base-patch32...
CLIP model loaded successfully
```

### Q4: 修改源码后更新 t2v_metrics 会怎么样？

如果通过 `pip install --upgrade t2v_metrics` 更新库，你的修改会被覆盖，需要重新修改。建议记录修改的内容或备份修改后的文件。

---

## 📚 参考资源

- [CLIP (HuggingFace)](https://huggingface.co/openai/clip-vit-base-patch32)
- [SigLIP (HuggingFace)](https://huggingface.co/google/siglip-so400m-patch14-384)
- [Aesthetic Predictor v2.5 (GitHub)](https://github.com/discus0434/aesthetic-predictor-v2-5)
- [Aesthetic Predictor v2.5 (PyPI)](https://pypi.org/project/aesthetic-predictor-v2-5/)
- [t2v_metrics (GitHub)](https://github.com/linzhiqiu/t2v_metrics)
- [clip-flant5-xxl (HuggingFace)](https://huggingface.co/zhiqiulin/clip-flant5-xxl)
- [clip-vit-large-patch14-336 (HuggingFace)](https://huggingface.co/openai/clip-vit-large-patch14-336)
