<div align="center">

# ✨ ASTERISM:<br>Do Multimodal LLMs Actually Look at All the Images?

**Chanseul Cho** · **Dongwoo Kim** · **Noori Bae** · **Seungjae Moon** · **Seongsu Lee** · **Youngmin Ro**

Machine Intelligence Lab, University of Seoul

[![Paper](https://img.shields.io/badge/Paper-coming%20soon-lightgrey)](#)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b)](#)
[![Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-coming%20soon-yellow)](#datasets)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-coming%20soon-yellow)](#models)

</div>

<p align="center">
  <img src="assets/teaser.png" alt="ASTERISM teaser" width="100%">
</p>

## News

- Code, datasets and model weights will be released soon.

## Overview

Multi-image benchmarks aim to test whether a model combines evidence across images, but many of their items can be answered from a single image. Across about 33k items from 19 public benchmarks, our audit and a two-stage single-image filter find that one image or the question alone settles 24.2% of them, and MLLMs take this single-image shortcut.

- **✨ ASTERISM** keeps the items that need more than one image: 18.7k items under one format, one rule-based scorer and seven task categories, with every corrected label and every rewrite checked by a human reviewer.
- **CICA** (cross-image credit assignment) extends RLVR. For every token of a response, it compares the policy's likelihood under all images with its likelihood under the best single image, and moves credit toward the tokens that need more than one image. It keeps the data, the reward and the RL objective, and adds no parameters, rollouts or backward pass.

## Release

### Datasets

| Dataset | Items | Description | Link |
|---|---:|---|:---:|
| ✨ ASTERISM | 18,696 | Multi-image items from 19 public benchmarks that need more than one image | Coming soon |
| ✨ ASTERISM-Lite | 400 | Hard subset of ASTERISM, verified one by one | Coming soon |
| ✨ ASTERISM-Train | 49,105 | Training items with verifiable answers, with near-duplicates of ASTERISM images removed | Coming soon |

### Models

| Model | Base model | ASTERISM (base → CICA) | Link |
|---|---|:---:|:---:|
| CICA-Qwen2.5-VL-7B | Qwen2.5-VL-7B-Instruct | 40.7 → 46.7 | Coming soon |
| CICA-GLM-4.1V-9B-Thinking | GLM-4.1V-9B-Thinking | 58.7 → 61.3 | Coming soon |
| CICA-Qwen3.5-9B | Qwen3.5-9B (thinking mode) | 65.7 → 70.6 | Coming soon |
| CICA-Qwen3.8-27B | Qwen3.8-27B | – | Coming soon |

ASTERISM scores are macro averages over the seven task categories.

## Code

Coming soon:

- `asterism/`: prompts, the rule-based scorer and evaluation runners for ASTERISM and ASTERISM-Lite
- `cica/`: CICA training with the settings of the paper
- `build/`: scripts that assemble the items of restricted sources from their public releases

## License

- Code: [Apache License 2.0](LICENSE).
- Our annotations (corrected labels and answer formats, rewritten items, per-item labels): CC BY 4.0.
- Source items and images keep the license of their benchmark. See [DATA_LICENSE.md](DATA_LICENSE.md).
- Model weights follow the license of their base models.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{asterism,
  title   = {ASTERISM: Do Multimodal LLMs Actually Look at All the Images?},
  author  = {Cho, Chanseul and Kim, Dongwoo and Bae, Noori and Moon, Seungjae and Lee, Seongsu and Ro, Youngmin},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {XXXX}
}
```
