# Reproducibility, citation and references

*Full report, chapter 6 of 6 (§11–§12).* [Overview](../README.md) · [Report contents](README.md)

---

## 11. Repository map and reproducibility

```
6Band-Multispectral-UQ/
├── README.md              ← overview: what we did and the end result
├── LICENSE                ← Apache 2.0
├── docs/                  ← full report: six chapters + figures/
│   ├── 01_…06_*.md        ←   the six chapters (README.md = contents)
│   └── figures/           ←   figures made for this documentation
│       ├── dataset/       ←   data analysis (Figures 1–10)
│       ├── uncertainty/   ←   cross-method results (Figures 12–23)
│       └── maps/          ←   uncertainty maps extracted from the result workbooks
├── baseline/  credit/  ensemble/  dapm/
├── multicp/   sacp/    multicp_sacp/
├── dofa/  georsclip/  mambahsi/  spatialgcn/
│   ├── README.md          ← method write-up: intuition, maths, results, caveats
│   ├── *.ipynb            ← executed notebooks
│   └── results/           ← CSV / JSON / PNG / XLSX outputs (the source of every number)
└── scripts/surveillance_monitor.py  ← helper that tracked the Kaggle kernel runs
```

| Resource | Link |
|---|---|
| Data (`data.csv`, `ref.csv`) | 🤗 [datasets/DaneshSelwal/Classification-6Bands-Data](https://huggingface.co/datasets/DaneshSelwal/Classification-6Bands-Data) |
| Trained weights | 🤗 [DaneshSelwal/Classification-6Bands-Models](https://huggingface.co/DaneshSelwal/Classification-6Bands-Models) |
| 65-band companion study | [DIAS-65Band-Hyperspectral-UQ](https://github.com/DaneshSelwal/DIAS-65Band-Hyperspectral-UQ) |
| Umbrella project | [Classification_Uncertainty_Quantification_Analysis](https://github.com/DaneshSelwal/Classification_Uncertainty_Quantification_Analysis) |

**Kaggle notebooks** (one per step; status from the repository's run log):

| Folder | Training | Uncertainty |
|---|---|---|
| baseline | [cuq6-baseline-k01](https://www.kaggle.com/code/daneshselwal/cuq6-baseline-k01) ✅ | [cuq6-uncertainty-k02](https://www.kaggle.com/code/daneshselwal/cuq6-uncertainty-k02) ✅ |
| credit | [cuq6-credit-k03](https://www.kaggle.com/code/daneshselwal/cuq6-credit-k03) ✅ | — |
| ensemble | [cuq6-ensemble-k04](https://www.kaggle.com/code/daneshselwal/cuq6-ensemble-k04) ⚠️ not re-run | [cuq6-crede-k05](https://www.kaggle.com/code/daneshselwal/cuq6-crede-k05) ✅ |
| multicp | [cuq6-multicp-k06](https://www.kaggle.com/code/daneshselwal/cuq6-multicp-k06) ⚠️ cancelled | [cuq6-multicp-uq-k07](https://www.kaggle.com/code/daneshselwal/cuq6-multicp-uq-k07) ❌ failed |
| multicp_sacp | — | [cuq6-multicp-sacp-k08](https://www.kaggle.com/code/daneshselwal/cuq6-multicp-sacp-k08) ❌ failed |
| sacp | — | [cuq6-sacp-k09](https://www.kaggle.com/code/daneshselwal/cuq6-sacp-k09) ✅ |
| mambahsi | [cuq6-mambahsi-k10](https://www.kaggle.com/code/daneshselwal/cuq6-mambahsi-k10) ✅ | [cuq6-mambahsi-uq-k11](https://www.kaggle.com/code/daneshselwal/cuq6-mambahsi-uq-k11) ✅ |
| dofa | [cuq6-dofa-k12](https://www.kaggle.com/code/daneshselwal/cuq6-dofa-k12) ✅ | [cuq6-dofa-uq-k13](https://www.kaggle.com/code/daneshselwal/cuq6-dofa-uq-k13) ✅ |
| georsclip | [cuq6-georsclip-k14](https://www.kaggle.com/code/daneshselwal/cuq6-georsclip-k14) ✅ | [cuq6-georsclip-uq-k15](https://www.kaggle.com/code/daneshselwal/cuq6-georsclip-uq-k15) ✅ |
| dapm | — (Colab only) | — (Colab only) |
| spatialgcn | [cuq6-spatialgcn-k16](https://www.kaggle.com/code/daneshselwal/cuq6-spatialgcn-k16) ✅ | [cuq6-spatialgcn-uq-k17](https://www.kaggle.com/code/daneshselwal/cuq6-spatialgcn-uq-k17) ✅ |

**To reproduce:** run each folder's training notebook, then its uncertainty notebook, on a GPU. Seeds are
fixed (42; AlexNet 10). The notebooks find `data.csv`/`ref.csv` (or `k001.csv`/`k002.csv`) under
`/kaggle/input`, `data/` or the working directory.

---

## 12. Citation, licence and references

**Licence.** [Apache License 2.0](../LICENSE).

```bibtex
@misc{selwal2026sixband_uq,
  author       = {Selwal, Danesh},
  title        = {Uncertainty Quantification for 6-Band Multispectral
                  Land-Cover Classification},
  year         = {2026},
  howpublished = {\url{https://github.com/DaneshSelwal/6Band-Multispectral-UQ}}
}
```

**References.**

1. Angelopoulos, A. N., & Bates, S. (2023). *Conformal Prediction: A Gentle Introduction.* Foundations and Trends in ML.
2. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
3. Angelopoulos, A. N., et al. (2021). *Uncertainty Sets for Image Classifiers using Conformal Prediction* (RAPS). ICLR.
4. Romano, Y., Sesia, M., & Candès, E. (2020). *Classification with Valid and Adaptive Coverage* (APS). NeurIPS.
5. Ding, T., et al. (2023). *Class-Conditional Conformal Prediction with Many Classes* (Clustered CP). NeurIPS.
6. Shi, Y., et al. (2024). *Conformal Prediction for Class-wise Coverage via Augmented Label Rank Calibration* (RC3P). NeurIPS.
7. Huang, J., et al. (2024). *Conformal Prediction for Deep Classifier via Label Ranking* (SAPS). ICML.
8. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Deep Ensembles.* NeurIPS.
9. Hüllermeier, E., & Waegeman, W. (2021). *Aleatoric and Epistemic Uncertainty in Machine Learning.* Machine Learning 110.
10. Ganin, Y., et al. (2016). *Domain-Adversarial Training of Neural Networks.* JMLR.
11. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
12. Rao, Y., et al. (2021). *Global Filter Networks for Image Classification.* NeurIPS.
13. Dosovitskiy, A., et al. (2021). *An Image is Worth 16×16 Words.* ICLR.
14. Xiong, Z., et al. (2024). *Neural Plasticity-Inspired Foundation Model for Observing the Earth Crossing Modalities* (DOFA). arXiv:2403.15356.
15. Ryali, C., et al. (2023). *Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles.* ICML. arXiv:2306.00989.
16. Li, Y., et al. (2024). *MambaHSI: Spatial-Spectral Mamba for Hyperspectral Image Classification.* arXiv:2404.18401.
17. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
18. Tucker, C. J. (1979). *Red and photographic infrared linear combinations for monitoring vegetation.* Remote Sensing of Environment 8 (NDVI).
19. Roberts, D. R., et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* Ecography 40.
20. Multi-CP reference implementation: [github.com/yamtawa/Multi-CP](https://github.com/yamtawa/Multi-CP).

---

[← Limitations and threats to validity](05_limitations.md) · [Report contents](README.md) · [Back to overview →](../README.md)
