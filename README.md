<div align="center">

# Uncertainty Quantification for 6-Band Multispectral Land-Cover Classification

### A Comparative Study of Conformal, Credal and Diffusion-Based Methods across 11 Deep Classifiers

[![License](https://img.shields.io/badge/license-Apache%202.0-2a78d6)](LICENSE)
![Bands](https://img.shields.io/badge/spectral%20bands-6-2a78d6)
![Scene](https://img.shields.io/badge/scene-330%C3%97307-2a78d6)
![Classes](https://img.shields.io/badge/classes-7-2a78d6)
![Backbones](https://img.shields.io/badge/backbones-11-eb6834)
![UQ methods](https://img.shields.io/badge/uncertainty%20methods-11-1baf7a)
![Framework](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-FF6F00?logo=tensorflow&logoColor=white)

<img src="docs/figures/dataset/scene_overview.png" width="100%" alt="Natural colour, false colour and reference labels of the 6-band scene">

</div>

A land-cover classifier labels every pixel, including the ones it should not be sure about. This project
asks **how well modern uncertainty methods tell us which pixels of a land-cover map to trust**. We train
11 deep classifiers on a 6-band multispectral scene, attach 11 uncertainty-quantification (UQ) methods to
them, and measure whether each method's promise holds.

> This page is the short version. The full write-up, with the data analysis, maths and every figure, is in
> **[`docs/`](docs/README.md)**, and each method folder has its own detailed README.

---

## What we did

| Step | What | Details |
|:-:|---|---|
| **1** | **Data.** A 330 × 307-pixel scene with 6 spectral bands and 7 land-cover classes (17,239 labelled pixels). Each pixel is classified from the 9 × 9 patch around it | [Data and set-up](docs/01_data_and_setup.md) |
| **2** | **Classifiers.** 11 backbones: a CNN (AlexNet-style), a Fourier network (GFNet), Transformers (ViT-UNet, Hiera), state-space models (MambaHSI S/B/L), a graph network (SpatialGCN) and foundation-style models (DOFA, GeoRSCLIP) | [Models and classification](docs/02_models_and_classification.md) |
| **3** | **Uncertainty.** 11 UQ methods on top: five conformal-prediction variants, spatial and multi-head conformal prediction, credal ensembles and their distilled student, and a diffusion model with a statistical test | [Methods in plain words](docs/03_uncertainty_methods.md) |
| **4** | **Evaluation.** Coverage of the 95% promise overall and per class, size and shape of the prediction sets, and how much of the map each method calls "uncertain" | [Uncertainty results](docs/04_uncertainty_results.md) |

The scene is a useful test because the models are **not** all perfect: accuracy ranges from 68.8% to
99.9%, so we can see how each uncertainty method behaves on both strong and weak models.

---

## End result

### Classification

**Nine of the 11 models are near-perfect (99.3–99.9%).** MambaHSI-Large is the best at 99.88%. The two
exceptions, **SpatialGCN (84.6%)** and **DOFA-Spectral (68.8%)**, are the most useful for testing
uncertainty methods, because they make real mistakes. Part of the near-perfect accuracy is optimistic,
because of how the pixels were split (finding F8).

### Uncertainty: eight findings

| # | Finding | Evidence |
|:-:|---|---|
| F1 | **The 95% guarantee holds.** Split, class-conditional, clustered and RC3P conformal prediction cover the true class 94.9–97.2% of the time on all 11 backbones | [§8.1](docs/04_uncertainty_results.md#81-rq1--does-the-95-guarantee-hold) |
| F2 | **What the answer looks like depends on model quality.** Accurate models give one class, or abstain (an empty set) on 3–5% of pixels. SpatialGCN lists 2+ classes for 38–53% of pixels, DOFA-Spectral for 74–78% | [§8.2](docs/04_uncertainty_results.md#82-rq2--what-do-the-prediction-sets-look-like) |
| F3 | **"95% on average" can hide a failing class.** Split CP covers AlexNet's rarest class only 73.5% of the time. Class-conditional CP brings it back to 95.6% | [§8.3](docs/04_uncertainty_results.md#83-rq3--is-coverage-fair-across-classes) |
| F4 | **RC3P over-delivers** (95.7–97.2%): safe, but its sets are larger than needed | [§8.1](docs/04_uncertainty_results.md#81-rq1--does-the-95-guarantee-hold) |
| F5 | **The method's rule decides how much of the map is "uncertain"**: 0.09–0.25% (DAPM), 9–13% (CREDIT, MultiCP), 32–38% (SCMCP, 9 × 9 window) | [§8.5](docs/04_uncertainty_results.md#85-rq5--how-much-of-the-map-does-each-method-call-uncertain) |
| F6 | **Requiring 7 heads to agree multiplies the misses.** MultiCP's joint coverage is 71.8%, close to 0.95⁷ = 69.8% | [§8.6](docs/04_uncertainty_results.md#86-rq6--multi-head-and-spatial-variants) |
| F7 | **The ensembles disagree mostly off the labelled area.** 98.2–99.7% of the pixels CREDIT flags are unlabelled | [§8.7](docs/04_uncertainty_results.md#87-rq7--what-do-ensembles-say-about-the-two-kinds-of-uncertainty) |
| F8 | **Accuracy is inflated by the random split.** Every test patch contains training pixels (40.2 of 81 on average) | [§4.3](docs/01_data_and_setup.md#43-how-pixels-are-split-and-why-it-matters) |

<p align="center"><img src="docs/figures/uncertainty/coverage_all_backbones.png" width="95%" alt="Coverage of the five baseline conformal methods on all 11 backbones"></p>

*Coverage of the five baseline conformal methods on every backbone. Dots inside the shaded band are
statistically indistinguishable from the 95% target. RAPS (bottom row) sits at the model's accuracy because,
as implemented, it always returns only the top class.*

### Which method to use

| Method | Verdict |
|---|---|
| **Class-conditional CP** | ✅ **Recommended default.** Valid overall *and* for every class (≥ 0.90), at no extra cost at test time |
| **RC3P** | ✅ Valid and fair, but slightly larger sets than needed |
| **Split CP** | ⚠️ Valid on average, but can under-cover rare classes (worst class 0.74) |
| **Clustered CP** | ⚠️ Valid on average; fairer than Split CP, but not for every class (worst 0.79) |
| **RAPS (as implemented)** | ❌ Coverage equals accuracy rather than 95% on accurate models |
| **SACP** | ⚠️ Spatially coherent maps; ~99.5% coverage, which comes from never allowing an empty set |
| **MultiCP / SCMCP** | ❌ ≈ 72% joint coverage, below the 95% target |
| **CreDE / CREDIT** | ✅ Best at showing *where the model has never been trained* (no formal guarantee) |
| **DAPM** | ❌ Flags only ~0.1% of the scene |

---

## Read more

| | Where |
|---|---|
| 📖 **Full report** (data, set-up, models, methods, results, limitations) | [`docs/`](docs/README.md) |
| Five conformal methods on AlexNet / GFNet / ViT-UNet | [`baseline/`](baseline/README.md) |
| Spatially-aware conformal prediction (SACP) | [`sacp/`](sacp/README.md) |
| Multi-head conformal prediction (MultiCP) and its smoothed version (SCMCP) | [`multicp/`](multicp/README.md) · [`multicp_sacp/`](multicp_sacp/README.md) |
| Credal ensembles (CreDE) and the distilled student (CREDIT) | [`ensemble/`](ensemble/README.md) · [`credit/`](credit/README.md) |
| Diffusion-based uncertainty (DAPM) | [`dapm/`](dapm/README.md) |
| Extra backbones, each with the five conformal methods | [`dofa/`](dofa/README.md) · [`georsclip/`](georsclip/README.md) · [`mambahsi/`](mambahsi/README.md) · [`spatialgcn/`](spatialgcn/README.md) |
| Limitations, read before using the numbers | [Chapter 5](docs/05_limitations.md) |

**Data and weights.** 🤗 [Dataset](https://huggingface.co/datasets/DaneshSelwal/Classification-6Bands-Data) ·
🤗 [Trained models](https://huggingface.co/DaneshSelwal/Classification-6Bands-Models) ·
[Kaggle notebooks and how to reproduce](docs/06_reproducibility_and_references.md#11-repository-map-and-reproducibility) ·
companion 65-band study: [DIAS-65Band-Hyperspectral-UQ](https://github.com/DaneshSelwal/DIAS-65Band-Hyperspectral-UQ)

---

## Licence and citation

Released under the [Apache License 2.0](LICENSE).

```bibtex
@misc{selwal2026sixband_uq,
  author       = {Selwal, Danesh},
  title        = {Uncertainty Quantification for 6-Band Multispectral
                  Land-Cover Classification},
  year         = {2026},
  howpublished = {\url{https://github.com/DaneshSelwal/6Band-Multispectral-UQ}}
}
```
