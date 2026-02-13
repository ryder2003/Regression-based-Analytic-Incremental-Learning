# Regression-based Analytic Incremental Learning (RAIL)

[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-blue.svg)](https://neurips.cc/Conferences/2024)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/pdf/2406.18868)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-orange.svg)](https://pytorch.org/)

Official implementation of **"Advancing Cross-domain Discriminability in Continual Learning of Vision-Language Models"** accepted at **NeurIPS 2024**.

## 📋 Overview

RAIL addresses continual learning challenges in vision-language models by introducing:
- **Primal RAIL**: Feature space adaptation with analytic learning
- **Dual RAIL**: Kernel ridge regression in dual space for efficient continual learning
- Minimal catastrophic forgetting across multiple visual domains
- Few-shot learning capability (16 shots per class)

### Key Features
- 🎯 **High Performance**: Achieves ~87% average accuracy across multiple tasks
- ⚡ **Efficient**: Analytic solutions with low computational overhead
- 🔄 **Continual Learning**: Supports multiple sequential tasks without forgetting
- 🎨 **Multi-Domain**: Tested on 10+ visual recognition datasets

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 6GB+ GPU memory

### Setup Steps

1. **Clone the repository:**
```bash
git clone https://github.com/ryder2003/Regression-based-Analytic-Incremental-Learning.git
cd Regression-based-Analytic-Incremental-Learning
```

2. **Create conda environment:**
```bash
conda create -n rail python=3.8
conda activate rail
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install CLIP:**
```bash
pip install git+https://github.com/openai/CLIP.git
```

## 📂 Data Preparation

### Directory Structure
```
Regression-based-Analytic-Incremental-Learning/
├── datasets/
│   ├── dtd/
│   ├── oxford_flowers/
│   ├── oxford_pets/
│   ├── caltech101/
│   ├── cifar100/
│   └── ...
├── configs/
├── clip/
└── scenario_datasets/
```

### Download Datasets

Please refer to [CoOp's dataset guide](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) for detailed instructions on downloading and preparing datasets.

**Supported Datasets:**
- DTD (Describable Textures Dataset)
- Oxford Flowers-102
- Oxford-IIIT Pets
- C📝 Citation

If you find this work helpful, please cite:

```bibtex
@article{xu2024advancing,
  title={Advancing Cross-domain Discriminability in Continual Learning of Vision-Language Models},
  author={Xu, Yicheng and Chen, Yuxin and Nie, Jiahao and Wang, Yusong and Zhuang, Huiping and Okumura, Manabu},
  journal={arXiv preprint arXiv:2406.18868},
  year={2024},
  note={Accepted at NeurIPS 2024}
}
```

## 🙏 Acknowledgements

This project builds upon excellent work from:
- [CLIP](https://github.com/openai/CLIP) - OpenAI's vision-language model
- [CoOp](https://github.com/KaiyangZhou/CoOp) - Context optimization for prompt learning

## 📧 Contact

For questions, discussions, or collaborations:
- **Email**: yxu040@e.ntu.edu.sg
- **WeChat**: linghan199

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) file for details.

---

<div align="center">
  <b>⭐ Star this repo if you find it helpful! ⭐</b>
</div>yaml
datasets: [dtd, oxford_flowers, oxford_pets]  # Dataset sequence
backbone: ViT-B/16                            # CLIP backbone
num_shots: 16                                 # Few-shot samples per class
batch_size: 64                                # Training batch size
hidden_dim: 2048                              # Feature expansion dimension
```

## 🏃 Running Experiments

### Primal RAIL
```bash
python primal_RAIL.py
```

### Dual RAIL
```bash
python dual_RAIL.py
```

### Test Setup
Verify your environment and datasets:
```bash
python test_gpu.py        # Check GPU availability
python test_datasets.py   # Verify dataset loading
python test_run.py        # Run quick test
```

## 📊 Results

### Performance on Sequential Tasks

| Method | DTD | Oxford Flowers | Oxford Pets | Average |
|--------|-----|----------------|-------------|---------|
| Primal RAIL | 71.70% | 94.39% | 95.95% | **87.35%** |
| Dual RAIL | - | - | - | - |

*Trained with 16 shots per class on NVIDIA RTX 3050 6GB*

## Citation
```bash
@article{xu2024advancing,
  title={Advancing Cross-domain Discriminability in Continual Learning of Vision-Language Models},
  author={Xu, Yicheng and Chen, Yuxin and Nie, Jiahao and Wang, Yusong and Zhuang, Huiping and Okumura, Manabu},
  journal={arXiv preprint arXiv:2406.18868},
  year={2024}
}
```
---

## Acknowledgement

Our repo benefits from [CLIP](https://github.com/openai/CLIP) and [CoOp](https://github.com/KaiyangZhou/CoOp). We thank them for their wonderful works.

---


