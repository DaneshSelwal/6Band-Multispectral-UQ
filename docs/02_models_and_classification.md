# Models and classification results

*Full report, chapter 2 of 6 (§5–§6).* [Overview](../README.md) · [Report contents](README.md)

---

## 5. Model architectures

Three **baseline backbones** are used by every UQ method. Eight **extended backbones** test whether the
conclusions survive a change of architecture.

```mermaid
flowchart TB
    subgraph AlexNet["AlexNet-CNN (local texture)"]
        a1["9×9×6 patch"] --> a2["5 × Conv 3×3"] --> a3["MaxPool"] --> a4["Dense layers + dropout"] --> a5["softmax (7)"]
    end
    subgraph GFNet["GFNet (global frequency filter)"]
        g1["9×9×6 patch"] --> g2["3×3 tokens → 9 tokens"] --> g3["FFT → learned filter → iFFT<br/>+ MLP, repeated"] --> g4["pool"] --> g5["softmax (7)"]
    end
    subgraph ViT["ViT-UNet (attention + skip links)"]
        v1["9×9×6 patch"] --> v2["3×3 tokens + CLS"] --> v3["Transformer layers<br/>with U-Net style skips"] --> v4["Dense head"] --> v5["softmax (7)"]
    end
```

| Model | Idea in one line | Folder |
|---|---|---|
| **AlexNet-CNN** | Stacked small convolutions learn local spatial-spectral texture | [`baseline/`](../baseline/README.md) |
| **GFNet** | A learnable filter in the Fourier domain mixes all tokens cheaply | [`baseline/`](../baseline/README.md) |
| **ViT-UNet** | A Vision Transformer whose layers are linked like a U-Net | [`baseline/`](../baseline/README.md) |
| **DOFA-Spectral, Hiera-ViT, DOFA+Hiera Fusion** | A wavelength-conditioned hypernetwork (DOFA), a hierarchical ViT (Hiera), and both together | [`dofa/`](../dofa/README.md) |
| **GeoRSCLIP-TextHead** | Image features matched by cosine similarity to one text embedding per class | [`georsclip/`](../georsclip/README.md) |
| **MambaHSI Small / Base / Large** | A selective state-space model scans the patch as a sequence | [`mambahsi/`](../mambahsi/README.md) |
| **SpatialGCN** | 81 pixel-nodes on a 4-neighbour graph, mixed by graph convolutions | [`spatialgcn/`](../spatialgcn/README.md) |

---

## 6. Classification results

| Model | Test accuracy | Cohen's κ | Macro-F1 | NLL | Brier | ECE (15 bins) | Train time |
|---|--:|--:|--:|--:|--:|--:|--:|
| MambaHSI-Large | **99.88%** | 0.9984 | 0.9989 | 0.0229 | 0.0025 | 0.0197 | 16.1 min |
| MambaHSI-Base | 99.84% | 0.9978 | 0.9987 | 0.0236 | 0.0028 | 0.0205 | 6.2 min |
| MambaHSI-Small | 99.74% | 0.9965 | 0.9974 | 0.0229 | 0.0054 | 0.0174 | 2.1 min |
| AlexNet-CNN | 99.70% | 0.9959 | 0.9955 | 0.0085 | 0.0044 | **0.0013** | 11.4 min |
| DOFA+Hiera Fusion | 99.70% | 0.9959 | 0.9966 | 0.0361 | 0.0054 | 0.0256 | 5.4 min |
| GFNet | 99.63% | 0.9950 | 0.9954 | 0.0228 | 0.0057 | 0.0089 | 12.4 min |
| GeoRSCLIP-TextHead | 99.63% | 0.9950 | 0.9952 | 0.0459 | 0.0074 | 0.0353 | 6.3 min |
| Hiera-ViT | 99.44% | 0.9925 | 0.9921 | 0.0527 | 0.0102 | 0.0342 | 4.9 min |
| ViT-UNet | 99.28% | 0.9902 | 0.9904 | 0.0643 | 0.0136 | 0.0388 | 21.6 min |
| **SpatialGCN** | **84.57%** | 0.7883 | 0.7825 | 0.4534 | 0.2242 | 0.0597 | 1.1 min |
| **DOFA-Spectral** | **68.84%** | 0.5678 | 0.6476 | 0.7916 | 0.4198 | 0.0504 | 0.9 min |

*__Table 1.__ Test-set classification (4,310 pixels). Sources: `*/results/classification_summary.csv`.*

**What Table 1 shows.**
- **Nine models are near-perfect (99.3–99.9%).** Part of this is the random split (§4.3).
- **Two models make real mistakes.** DOFA-Spectral (68.8%) averages its patch into a single spectrum and
  so cannot separate the overlapping classes of Figure 7. SpatialGCN (84.6%) over-smooths: its maps show
  rounded blobs instead of rectangular parcels ([spatialgcn/](../spatialgcn/README.md)).
- **Calibration differs even among accurate models.** AlexNet's ECE (0.0013) is 30× lower than ViT-UNet's
  (0.0388). The transformer-style models are systematically *under*-confident, a typical effect of label
  smoothing.

| Ground truth | AlexNet-CNN | GFNet | ViT-UNet |
|:-:|:-:|:-:|:-:|
| <img src="../baseline/results/scene_visualizations/ground_truth_label_map.png" alt="gt"> | <img src="../baseline/results/scene_visualizations/AlexNet_CNN_initial_classification.png" alt="AlexNet"> | <img src="../baseline/results/scene_visualizations/GFNet_initial_classification.png" alt="GFNet"> | <img src="../baseline/results/scene_visualizations/ViT_UNet_initial_classification.png" alt="ViT"> |

*__Figure 11.__ Full-scene predictions of the baselines (notebook numbering 0–6).*

**Reading Figure 11.** Inside the labelled centre the three models agree. In the unlabelled 83% of the scene
they draw the same parcel structure but sometimes assign different classes. That disagreement cannot show
up in any accuracy number, and it is exactly what the uncertainty maps of §8.4 and §8.7 reveal.

---

[← Data and experimental set-up](01_data_and_setup.md) · [Report contents](README.md) · [Uncertainty methods in plain words →](03_uncertainty_methods.md)
