# MambaHSI: state-space sequence models at three sizes

[← back to main README](../README.md) · conformal methods explained in [baseline/](../baseline/README.md#3-the-five-methods-step-by-step)

| | |
|---|---|
| **What** | A MambaHSI-style spatial-spectral state-space model in three capacities (Small / Base / Large), evaluated with the five baseline conformal methods |
| **Key result** | The most accurate models in the study: **99.74% / 99.84% / 99.88%**. Calibrated conformal methods cover 0.954–0.969 with single-class or empty (3.1–4.6%) sets |
| **The lesson** | Tripling model capacity changes accuracy by 0.14 points and leaves the uncertainty picture unchanged. Split CP's worst class still drops to **0.838** (Large, label 7); class-conditional CP restores ≥ 0.926 |
| **Notebooks** | [`Model_training_mambahsi.ipynb`](Model_training_mambahsi.ipynb) · [`Model_uncertainty_comparison_mambahsi.ipynb`](Model_uncertainty_comparison_mambahsi.ipynb) |
| **Paper in this folder** | [`2404.18401v3.pdf`](2404.18401v3.pdf) (MambaHSI, Li et al. 2024) |

---

## 1. The model in plain words

A **state-space model** reads a sequence one element at a time while carrying a compact memory (the
*state*). "Selective" models like Mamba decide, from the input itself, how much of the memory to keep and
how much new information to write:

```math
h_t = \bar A(x_t)\odot h_{t-1} + \bar B(x_t)\, x_t, \qquad y_t = C\, h_t
```

*In words: the new memory $`h_t`$ is the old memory, partly forgotten (gate $`\bar A`$), plus the current
token, partly admitted (gate $`\bar B`$). Both gates depend on the token, so the model can remember a
field's interior and "reset" at an edge.*

Here the 9 × 9 × 6 patch is cut into nine 3 × 3 tokens. Each token is projected to the hidden size, and
the sequence is scanned **forwards and backwards** (patches have no natural direction), then pooled and
classified.

| Variant | SSM blocks | State dim | Hidden dim | MLP ratio | Train time |
|---|--:|--:|--:|--:|--:|
| Small | 2 | 64 | 128 | 2 | 2.1 min |
| Base | 4 | 128 | 256 | 4 | 6.2 min |
| Large | 6 | 192 | 384 | 4 | 16.1 min |

> **Implementation note (from the notebook).** TensorFlow has no fused selective-scan kernel. The
> notebook therefore approximates Mamba with a diagonal linear recurrence whose gates are computed from the
> input at every step (`tf.scan`). It is faithful in spirit but **not a bit-exact Mamba**, and it is
> trained from scratch.

---

## 2. Results

| Model | Accuracy | κ | Macro-F1 | NLL | Brier | ECE |
|---|--:|--:|--:|--:|--:|--:|
| MambaHSI-Small | 99.74% | 0.9965 | 0.9974 | 0.0229 | 0.0054 | 0.0174 |
| MambaHSI-Base | 99.84% | 0.9978 | 0.9987 | 0.0236 | 0.0028 | 0.0205 |
| MambaHSI-Large | **99.88%** | 0.9984 | 0.9989 | 0.0229 | 0.0025 | 0.0197 |

*__Table 1.__ Source: [`results/classification_summary.csv`](results/classification_summary.csv).*

| Model | Method | Coverage | Avg. set size | Singletons | Empty | 2+ classes | Mean per-class cov. |
|---|---|--:|--:|--:|--:|--:|--:|
| **MambaHSI Base** | Split CP | 0.954 | 0.95 | 95.4% | 4.6% | 0.0% | 0.939 |
|  | Class-cond. CP | 0.966 | 0.97 | 96.6% | 3.4% | 0.0% | 0.967 |
|  | Clustered CP | 0.965 | 0.96 | 96.5% | 3.5% | 0.0% | 0.964 |
|  | RC3P | 0.968 | 0.97 | 96.8% | 3.2% | 0.0% | 0.970 |
|  | RAPS | 0.999 | 1.00 | 100.0% | 0.0% | 0.0% | 0.999 |
| **MambaHSI Large** | Split CP | 0.961 | 0.96 | 96.1% | 3.9% | 0.0% | 0.940 |
|  | Class-cond. CP | 0.965 | 0.96 | 96.5% | 3.5% | 0.0% | 0.962 |
|  | Clustered CP | 0.961 | 0.96 | 96.1% | 3.9% | 0.0% | 0.954 |
|  | RC3P | 0.969 | 0.97 | 96.9% | 3.1% | 0.0% | 0.967 |
|  | RAPS | 0.999 | 1.00 | 100.0% | 0.0% | 0.0% | 0.999 |
| **MambaHSI Small** | Split CP | 0.956 | 0.96 | 95.6% | 4.4% | 0.0% | 0.939 |
|  | Class-cond. CP | 0.961 | 0.96 | 96.1% | 3.9% | 0.0% | 0.964 |
|  | Clustered CP | 0.959 | 0.96 | 95.9% | 4.1% | 0.0% | 0.951 |
|  | RC3P | 0.968 | 0.97 | 96.8% | 3.2% | 0.0% | 0.971 |
|  | RAPS | 0.998 | 1.00 | 100.0% | 0.0% | 0.0% | 0.998 |

*__Table 2.__ Source: [`results/uncertainty_results/summary_metrics.csv`](results/uncertainty_results/summary_metrics.csv).*

| Model | Method | Label 1 | Label 2 | Label 3 | Label 4 | Label 5 | Label 6 | Label 7 | Worst |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| *eval pixels* | | *914* | *454* | *358* | *156* | *104* | *101* | *68* | |
| **MambaHSI Base** | Split CP | 0.985 | 0.952 | 0.902 | 0.936 | 0.923 | 0.960 | 0.912 | 0.902 |
|  | Class-cond. CP | 0.972 | 0.974 | 0.944 | 0.955 | 0.971 | 0.950 | 1.000 | 0.944 |
|  | Clustered CP | 0.970 | 0.974 | 0.944 | 0.955 | 0.971 | 0.931 | 1.000 | 0.931 |
|  | RC3P | 0.972 | 0.974 | 0.950 | 0.955 | 0.971 | 0.970 | 1.000 | 0.950 |
|  | RAPS | 0.999 | 1.000 | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 | 0.994 |
| **MambaHSI Large** | Split CP | 0.987 | 0.963 | 0.933 | 0.917 | 0.990 | 0.950 | **0.838** | 0.838 |
|  | Class-cond. CP | 0.969 | 0.965 | 0.950 | 0.974 | 1.000 | 0.950 | 0.926 | 0.926 |
|  | Clustered CP | 0.987 | 0.943 | 0.911 | 0.974 | 1.000 | 0.950 | 0.912 | 0.911 |
|  | RC3P | 0.973 | 0.967 | 0.958 | 0.994 | 1.000 | 0.950 | 0.926 | 0.926 |
|  | RAPS | 0.999 | 1.000 | 0.997 | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 |
| **MambaHSI Small** | Split CP | 0.985 | 0.941 | 0.925 | 0.955 | 0.952 | 0.950 | **0.868** | 0.868 |
|  | Class-cond. CP | 0.963 | 0.956 | 0.950 | 1.000 | 0.962 | 0.950 | 0.971 | 0.950 |
|  | Clustered CP | 0.967 | 0.956 | 0.950 | 0.974 | 0.913 | 0.970 | 0.926 | 0.913 |
|  | RC3P | 0.966 | 0.963 | 0.964 | 1.000 | 0.981 | 0.950 | 0.971 | 0.950 |
|  | RAPS | 0.999 | 0.998 | 0.992 | 1.000 | 1.000 | 1.000 | 1.000 | 0.992 |

*__Table 3.__ Source: [`results/uncertainty_results/per_class_coverage.csv`](results/uncertainty_results/per_class_coverage.csv).*

**Reading Tables 1–3.**
- **Capacity barely matters.** From Small to Large, parameters grow several-fold, accuracy rises 0.14
  points (about 6 pixels of 4,310), and NLL stays at 0.023. The task is saturated under this split.
- **Coverage is the baseline story again.** Split CP's empty-set rate (3.9–4.6%) is the error budget being
  spent on low-confidence correct pixels. RC3P is the most conservative (0.968–0.969).
- **The smallest class is the weak spot.** Split CP covers label 7 at 0.838 (Large) and 0.868 (Small).
  Better accuracy does not fix this, because Split CP never looks at classes. Class-conditional CP and RC3P
  lift every class of every variant to ≥ 0.926.

<p align="center"><img src="results/training_plots/model_comparison_metrics.png" width="90%" alt="comparison"></p>

*__Figure 1.__ The notebook's comparison of the three variants.*

<p align="center"><img src="results/scene_visualizations/combined_initial_classification_overview.png" width="100%" alt="maps"></p>

*__Figure 2.__ Ground truth and full-scene predictions of the three variants.*

## 3. Caveats

- Simplified, from-scratch Mamba (see the note above).
- With accuracy this close to 100%, differences between variants are within run-to-run noise. Only one
  seed was run.
- Random-split optimism ([main README §10](../README.md#10-limitations-and-threats-to-validity)).

## References

Gu & Dao (2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*; Li et al. (2024),
*MambaHSI*; Smith et al. (2023), *S5*.
