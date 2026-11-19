# GeoRSCLIP-style text head: classifying by similarity to class descriptions

[← back to main README](../README.md) · conformal methods explained in [baseline/](../baseline/README.md#3-the-five-methods-step-by-step)

| | |
|---|---|
| **What** | A CLIP-style classifier: a Hiera image tower and a small text tower are trained so that each patch lands close to the description of its class, evaluated with the five baseline conformal methods |
| **Key result** | **99.63%** accurate (ECE 0.035). Conformal coverage 0.951–0.965; sets are single-class or empty (3.5–4.9%) |
| **The lesson** | The text head is architecturally interesting but, with placeholder prompts, behaves like any other near-perfect backbone. Split CP's worst class (label 3) is covered 0.885; class-aware methods restore ≥ 0.95 |
| **Notebooks** | [`Model_training.ipynb`](Model_training.ipynb) · [`Model_uncertainty_comparison.ipynb`](Model_uncertainty_comparison.ipynb) |

---

## 1. The model in plain words

CLIP-style models do not learn a fixed "output neuron per class". Instead, they embed **the image** and a
**text description of each class** into the same space, then pick the class whose description is most
similar:

```math
\hat p_k(x) = \operatorname{softmax}_k\!\left(\frac{\langle f(x),\, g(t_k)\rangle}{\tau}\right),
\qquad \lVert f(x)\rVert = \lVert g(t_k)\rVert = 1
```

*In words: $`f(x)`$ is the image embedding of the patch, $`g(t_k)`$ the embedding of class $`k`$'s prompt
$`t_k`$, both scaled to length 1. Their dot product is the cosine similarity, and a temperature $`\tau`$
turns similarities into probabilities.*

| Setting | Value (from `GEORSCLIP_CFG`) |
|---|---|
| Image tower | Hiera, stage dims [128, 256], blocks [2, 2], 4 heads, MLP ratio 2 |
| Shared embedding | 128 dimensions |
| Text tower | small, trainable, embedding 64, prompts up to 12 tokens |
| Recipe | as baseline (100 epochs, batch 128, lr 3 × 10⁻⁴, seed 42); 6.3 min |

> ⚠️ **Not the pretrained GeoRSCLIP model.** No pretrained CLIP/GeoRSCLIP checkpoint was available in the
> run environment, so both towers are trained from scratch. The prompts are **placeholders**
> (`"a satellite photo of land cover class one"`, …, `"… class seven"`). The notebook itself notes that
> they "carry no real semantic signal". In practice the text tower therefore learns one free embedding per
> class, and the model behaves like a Hiera classifier with a cosine-similarity head. Re-running with real
> class names (once known) is the natural next experiment.

---

## 2. Results

| Accuracy | κ | Macro-F1 | NLL | Brier | ECE |
|--:|--:|--:|--:|--:|--:|
| 99.63% | 0.9950 | 0.9952 | 0.0459 | 0.0074 | 0.0353 |

*__Table 1.__ Source: [`results/classification_summary.csv`](results/classification_summary.csv).*

| Model | Method | Coverage | Avg. set size | Singletons | Empty | 2+ classes | Mean per-class cov. |
|---|---|--:|--:|--:|--:|--:|--:|
| **GeoRSCLIP TextHead** | Split CP | 0.951 | 0.95 | 95.1% | 4.9% | 0.0% | 0.950 |
|  | Class-cond. CP | 0.955 | 0.96 | 95.5% | 4.5% | 0.0% | 0.967 |
|  | Clustered CP | 0.954 | 0.95 | 95.4% | 4.6% | 0.0% | 0.961 |
|  | RC3P | 0.965 | 0.97 | 96.5% | 3.5% | 0.0% | 0.977 |
|  | RAPS | 0.998 | 1.00 | 100.0% | 0.0% | 0.0% | 0.997 |

*__Table 2.__ Source: [`results/uncertainty_results/summary_metrics.csv`](results/uncertainty_results/summary_metrics.csv).*

| Model | Method | Label 1 | Label 2 | Label 3 | Label 4 | Label 5 | Label 6 | Label 7 | Worst |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| *eval pixels* | | *914* | *454* | *358* | *156* | *104* | *101* | *68* | |
| **GeoRSCLIP TextHead** | Split CP | 0.966 | 0.963 | **0.885** | 0.968 | 0.990 | 0.921 | 0.956 | 0.885 |
|  | Class-cond. CP | 0.945 | 0.958 | 0.950 | 0.981 | 0.990 | 0.960 | 0.985 | 0.945 |
|  | Clustered CP | 0.944 | 0.963 | 0.950 | 0.962 | 0.990 | 0.960 | 0.956 | 0.944 |
|  | RC3P | 0.952 | 0.963 | 0.978 | 0.981 | 1.000 | 0.980 | 0.985 | 0.952 |
|  | RAPS | 0.999 | 0.998 | 0.994 | 1.000 | 1.000 | 0.990 | 1.000 | 0.990 |

*__Table 3.__ Source: [`results/uncertainty_results/per_class_coverage.csv`](results/uncertainty_results/per_class_coverage.csv).*

**Reading Tables 2–3.**
- Coverage behaves exactly as for the baseline backbones: Split, clustered and class-conditional CP
  0.951–0.955, RC3P 0.965, and RAPS equals accuracy (0.998, [why](../baseline/README.md#raps)).
- **Label 3 is Split CP's weak spot (0.885).** It is also the class most baseline models confuse with
  labels 2 and 4. Class-conditional CP and RC3P lift it to 0.950 and 0.978.
- ECE (0.035) is four times GFNet's (0.009): the learned temperature (`logit_scale`, initialised at
  ln 100 as in CLIP) leaves the probabilities less well calibrated. Conformal coverage is unaffected,
  because the cut-off is learned from the model's actual scores.

| Training curves | Confusion matrix |
|:-:|:-:|
| <img src="results/training_plots/GeoRSCLIP_TextHead_training_curves.png" alt="curves"> | <img src="results/training_plots/confusion_matrices_side_by_side.png" alt="cm"> |

*__Figure 1.__ Training curves and test confusion matrix.*

<p align="center"><img src="results/scene_visualizations/combined_initial_classification_overview.png" width="80%" alt="maps"></p>

*__Figure 2.__ Ground truth and full-scene prediction.*

## 3. Caveats

- Placeholder prompts and no pretraining (see the box above).
- Random-split optimism ([main README §10](../docs/05_limitations.md#10-limitations-and-threats-to-validity)).

## References

Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP);
Zhang et al. (2024), *RS5M and GeoRSCLIP*; Ryali et al. (2023), *Hiera*.
