# Limitations and threats to validity

*Full report, chapter 5 of 6 (§10).* [Overview](../README.md) · [Report contents](README.md)

---

## 10. Limitations and threats to validity

| # | Issue | Effect | Where |
|:-:|---|---|---|
| 1 | **Random pixel split**: every test patch contains training pixels (40.2 of 81 on average) | Accuracy and coverage are optimistic for new fields or scenes | [§4.3](01_data_and_setup.md#43-how-pixels-are-split-and-why-it-matters) |
| 2 | Small classes: class 7 has 68 evaluation pixels | Per-class coverage carries ±5 points of noise | [§3.2](01_data_and_setup.md#32-the-scene-and-its-classes) |
| 3 | AlexNet trained on split seed 10; conformal calibration uses seed 42 | **74.5%** of AlexNet's calibration/evaluation pixels were in its training set | [baseline/](../baseline/README.md#7-caveats) |
| 4 | RAPS implemented without randomisation | Collapses to top-1 on accurate models | [baseline/](../baseline/README.md#raps) |
| 5 | MultiCP intersects independently calibrated heads | Joint coverage ≈ 0.72 | [multicp/](../multicp/README.md) |
| 6 | MultiCP / SCMCP maps mark every **label-7** pixel as uncertain (`y == 7`) | Real class-7 parcels are greyed out regardless of the model | [multicp/](../multicp/README.md#33-maps) |
| 7 | MultiCP, SCMCP and DAPM results come from an earlier Google Colab run; the Kaggle run log marks the MultiCP (k06, k07), SCMCP (k08) and ensemble-training (k04) kernels as cancelled or failed | Those results cannot be regenerated from the committed notebooks alone without the original weights | [multicp/](../multicp/README.md), [multicp_sacp/](../multicp_sacp/README.md), [dapm/](../dapm/README.md), [ensemble/](../ensemble/README.md) |
| 8 | CREDIT selects its checkpoint on the test set | Slight optimism in CREDIT's metrics | [credit/](../credit/README.md) |
| 9 | Band wavelengths and class names are inferred, not documented | DOFA's wavelength conditioning and the class interpretations are approximate | [§3.1](01_data_and_setup.md#31-what-the-files-contain) |
| 10 | SACP / SCMCP smooth across calibration and test pixels | Exchangeability is bent; the formal guarantee does not strictly apply | [sacp/](../sacp/README.md#4-caveats) |
| 11 | Several "foundation-style" models are simplified and trained from scratch: DOFA without pretraining, GeoRSCLIP with placeholder prompts, MambaHSI with an approximate selective scan | They test the architectural idea, not the published pretrained models | [dofa/](../dofa/README.md), [georsclip/](../georsclip/README.md), [mambahsi/](../mambahsi/README.md) |
| 12 | One scene, one split, no repeated seeds | No confidence intervals across runs | — |

---

[← Uncertainty results and discussion](04_uncertainty_results.md) · [Report contents](README.md) · [Reproducibility, citation and references →](06_reproducibility_and_references.md)
