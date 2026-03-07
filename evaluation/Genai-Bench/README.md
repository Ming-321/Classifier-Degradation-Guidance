# GenAI-Bench Dataset

## Overview

This directory contains prompts from the **GenAI-Bench** dataset, used for evaluating text-to-image generation models on complex compositional prompts.

## Data Source

- **Original Dataset**: [GenAI-Bench](https://huggingface.co/datasets/TIGER-Lab/GenAI-Bench)
- **Provided by**: TIGER-Lab
- **HuggingFace Link**: https://huggingface.co/datasets/TIGER-Lab/GenAI-Bench
- **DataViewer**: https://huggingface.co/spaces/BaiqiL/GenAI-Bench-DataViewer

## License

The GenAI-Bench dataset is distributed under the **CC-BY-4.0** (Creative Commons Attribution 4.0 International) license.

**License Summary**:
- ✅ Commercial use allowed
- ✅ Modifications allowed
- ✅ Distribution allowed
- ⚠️ Must provide attribution to original authors
- ⚠️ Must indicate if changes were made

For full license text, see: https://creativecommons.org/licenses/by/4.0/

## Usage in This Project

The prompts are pre-processed and stored in `Genai-Bench.json` for use in our evaluation experiments. We use this dataset to evaluate the performance of our Classifier Degradation Guidance (CDG) method and compare it with baseline guidance methods.

**Experiments using GenAI-Bench**:
- `experiment/Genai-SD3/`: SD3 model evaluation
- `experiment/Genai-SD3.5/`: SD3.5 model evaluation  
- `experiment/Genai-Flux/`: FLUX model evaluation

## Attribution

```
@misc{genaibench2024,
  title={GenAI-Bench: A Holistic Benchmark for Generative AI},
  author={GenAI-Bench Team},
  howpublished={https://huggingface.co/datasets/TIGER-Lab/GenAI-Bench},
  year={2024}
}
```

If you use GenAI-Bench in your research, please cite the original dataset in addition to our paper.
