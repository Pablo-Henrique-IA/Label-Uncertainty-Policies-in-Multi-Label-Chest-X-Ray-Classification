# Label Uncertainty Policies in Multi-Label Chest X-Ray Classification

> **A Systematic Ablation Study (M1 → M2 → M3 → M4)**
> Xception 512×512 · Domain Transfer NIH → CheXpert · Clinical Uncertainty Policies · Grad-CAM + MedSAM

A four-step ablation study that **isolates one variable per step** — domain transfer, preprocessing, and uncertain-label policy — in a multi-label chest X-ray (CXR) classifier. Built on an Xception backbone fine-tuned from **NIH ChestX-ray14** onto **Stanford CheXpert** at 512×512, with a zero-shot interpretability layer (Grad-CAM → Grad-CAM++ → MedSAM).

**Projeto Integrador 2025/2026 — Faculdade SENAI FATESG** · Target venue: **ENIAC 2026 (BRACIS)**

<!-- Optional badges — uncomment / adjust as needed
![Python](https://img.shields.io/badge/python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)
-->

---

## TL;DR — Key Findings

1. **Domain transfer dominates.** NIH → CheXpert fine-tuning yields **+0.1015 Macro-AUROC** (M1→M2), ~87% of the total study gain and an order of magnitude larger than any uncertainty-policy effect (ratio ≈ 3.3:1).
2. **Unrestricted U-Ones collapses specificity.** Converting *all* uncertain labels to positive (M3) drives macro-specificity to **0.21** and Brier Score to **0.305 (+64% relative)** — clinically unusable for general screening.
3. **Selective U-Ones helps technically-uncertain labels.** A label-selective hybrid (M4) reaches **AUROC 0.9652 for pneumothorax** (+0.0394 vs. M2).
4. **Uncertainty policies are non-modular.** In a shared-backbone network, changing the policy for two labels degrades the *nominally-protected* labels too — a pattern consistent with indirect gradient interference.

---

## Ablation Design

Each model differs from the previous one in **exactly one** variable.

| Model | Init | Preprocessing | Uncertainty policy / ε | Isolated variable |
|-------|------|---------------|------------------------|-------------------|
| **M1** | ImageNet | — | — | baseline (NIH, 6 labels) |
| **M2** | M1 | Center Crop | U-Ignore | domain transfer |
| **M3** | M2 | Center Crop | U-Ones / 0.05 | uncertainty policy |
| **M4** | M3 | Center Crop | Hybrid / 0.15† | label selectivity |

† U-Ones (ε = 0.15) applied **only** to `pleural_effusion` and `pneumothorax`; U-Ignore for the other four labels.

The six labels (intersection of NIH and CheXpert): `atelectasis`, `cardiomegaly`, `pleural_effusion`, `pneumothorax`, `consolidation`, `edema`.

---

## Results

### Macro metrics (CheXpert official validation set, n = 159, threshold = 0.5)

| Model | Macro-AUROC | Macro-AUPRC | F1 | Sensitivity | Specificity | Brier ↓ |
|-------|:-----------:|:-----------:|:--:|:-----------:|:-----------:|:-------:|
| M1 | 0.7697 | — | — | — | — | — |
| **M2** (publication) | **0.8712** | **0.7209** | **0.6191** | 0.8893 | **0.5835** | **0.186** |
| M3 | 0.8402 | 0.6599 | 0.4882 | 0.9974\* | 0.2103 | 0.305 |
| M4 | 0.8548 | 0.6719 | 0.5449 | 0.9741 | 0.4124 | 0.224 |

\* M3's near-perfect sensitivity is a **positive-prediction bias**, not a clinical gain (specificity collapse). ΔBrier vs. M2: M3 +63.9%, M4 +20.4%.

### Per-label AUROC

| Label | M1 | M2 | M3 | M4 | Best |
|-------|:--:|:--:|:--:|:--:|:----:|
| atelectasis | 0.7044 | **0.8106** | 0.7970 | 0.7898 | M2 |
| cardiomegaly | 0.8114 | **0.8252** | 0.7714 | 0.8166 | M2 |
| pleural_effusion ⋆ | 0.7858 | **0.8776** | 0.8684 | 0.8388 | M2 |
| pneumothorax ⋆ | 0.8396 | 0.9258 | 0.9314 | **0.9652** | M4 |
| consolidation | 0.6493 | **0.8932** | 0.8595 | 0.8526 | M2 |
| edema | 0.8277 | **0.8946** | 0.8136 | 0.8657 | M2 |
| **Macro** | 0.7697 | **0.8712** | 0.8402 | 0.8548 | M2 |

⋆ = label under U-Ones in M4. M2 dominates 5/6 labels; M4 contributes the single best individual result (pneumothorax).

> **Statistical note.** DeLong tests (M4 vs. M2) reach no significance (p > 0.30 for all labels): n = 159 gives power < 30% to detect ΔAUROC ≈ 0.04. Pneumothorax has only n⁺ = 7 positives — its 0.9652 is reported as an **ablation finding**, not a proven gain (95% CI bootstrap [0.88–1.00]). Deltas should be read accordingly.

---

## Datasets

| | NIH ChestX-ray14 | Stanford CheXpert |
|---|---|---|
| Role | Domain pre-training (M1) | Domain fine-tuning (M2–M4) |
| Frontal images | 112,120 | 191,027 (full) |
| Patients | 30,805 | 64,540 |
| Resolution | 1024×1024 (uniform) | variable (median ≈ 2828×2320) |
| Labels | 14 (binary) | 14 (ternary +1 / 0 / −1) |
| Used here | 41,086 train / 20,486 test | 87,258 fine-tuning / 159 validation |

Splits are **patient-level**; contamination verified = 0. Only frontal images are used (eliminates aspect-ratio variation across projections). CheXpert ternary labels encode clinical uncertainty (−1), the central variable of the M2→M4 ablation.

**Uncertain-label prevalence (CheXpert, fine-tuning subset):** atelectasis 38.7%, consolidation 31.8%, edema 14.9%, pleural effusion 13.3%, cardiomegaly 3.3%, pneumothorax 2.9%. The two labels chosen for selective U-Ones in M4 (pneumothorax, pleural effusion) have the *lowest* and *technically-driven* uncertainty.

> **Datasets are not redistributed here.** Download NIH ChestX-ray14 and CheXpert from their official sources and accept their respective licenses/usage terms. This repo ships **code, configs, and CSV manifests** (paths + labels + Focal-Loss weights + separate U-Ones/U-Ignore columns), not raw images.

---

## Architecture

- **Backbone:** Xception (ImageNet pre-trained). Depthwise separable convolutions cut per-block cost from 240 → 63 operations while preserving subtle textural detail at 512×512 (critical for pneumothorax: absence of peripheral vascular marking).
- **Input:** 512×512×3 (grayscale replicated to 3 channels; normalized µ = 0.5056, σ = 0.2520).
- **Head:** `GAP → Dense(512)+BN+Dropout(0.5) → Dense(256)+Dropout(0.3) → Dense(6, sigmoid)`.
- **Parameters:** 22,045,486.
- Grad-CAM extracted from `block14_sepconv2_act`.

### Training protocol — 3-phase progressive fine-tuning

| Phase | Backbone | LR | Epochs | Rationale |
|-------|----------|----|--------|-----------|
| 1 | frozen | 1e-3 | 15 | adapt head to new domain |
| 2 | top 30% unfrozen | 1e-4 | 40 | adjust high-level blocks |
| 3 | fully unfrozen | 1e-5 | 30 | fine refinement |

**Common config:** Adam · batch = 32 · seed = 42 · BatchNorm frozen across all phases (preserves ImageNet stats under the noisy small-batch regime) · light augmentation (brightness 0.1, contrast 0.9–1.1) · ReduceLROnPlateau (factor 0.5, patience 3) · EarlyStopping (patience 7, monitor `val_macro_auroc`).

**Loss:** Focal Loss (γ = 2.0, α = 0.25) with per-label weights `w_ℓ = N / (L · n⁺_ℓ)`, where `n⁺_ℓ` is the effective positive count under the active uncertainty policy.

### Preprocessing decision — Center Crop vs. Zero-Padding

Both strategies were **trained and compared empirically** (not chosen by assumption). Center Crop won 5/6 labels and Macro-AUROC (**0.8710 vs. 0.8562**); the only Padding win was `pleural_effusion` (costophrenic angles preserved). The largest gap was pneumothorax (ΔAUROC +0.0442) — Padding's artificial black borders visually mimic peripheral hypertransparency, a likely spurious attention artifact. **Center Crop adopted for M2–M4.**

---

## Interpretability — Grad-CAM → Grad-CAM++ → MedSAM (zero-shot)

A no-annotation, no-fine-tuning localization pipeline:

1. **Grad-CAM / Grad-CAM++** generate the saliency map of the highest-confidence label over Xception at 512×512.
2. The Grad-CAM++ map is **binarized into an automatic bounding box**.
3. The box prompts **MedSAM** (SAM medical adaptation, trained on 1,570,263 image–mask pairs across 10 modalities), which returns an anatomical segmentation mask.

Both saliency maps are shown side by side (thoracic pathologies are frequently multifocal — Grad-CAM++ covers all foci, Grad-CAM highlights the dominant one). Segmentation is currently **proof-of-concept / qualitative only** — IoU/Dice against annotated boxes is future work.

---

## Application — CXR Analyzer

An interactive **Gradio** app packaging the publication model (**M2 Center Crop U-Ignore**): adjustable threshold, per-pathology probabilities, and four panels (Original · Grad-CAM · Grad-CAM++ · MedSAM) for the highest-confidence finding. <!-- TODO: add screenshot + live demo / Hugging Face Space link if available -->

---

## Limitations

- **Validation power.** n = 159 → statistical power < 30% for ΔAUROC ≈ 0.04; pneumothorax (n⁺ = 7) is especially unstable.
- **Sequential design.** M1→M4 prevents isolated factorial causal attribution; the M4 regression may reflect gradient interference, inherited weight degradation from M3, or both.
- **Policy/loss coupling.** Per-label Focal-Loss weights depend on `n⁺_ℓ`, which shifts with the policy — switching policies alters both label semantics and effective class weights simultaneously.
- **Single seed** per model — training variance not estimated.
- **No external validation** (MIMIC-CXR, PadChest) yet.
- **Partial policy coverage** — U-Zeros, U-SelfTrained, U-MultiClass remain future comparisons.
- **Segmentation** evaluated qualitatively only (IoU/Dice pending).

## Future work

Control with fixed `w_ℓ` across policies (decouple semantics from reweighting) · factorial ablation initializing M4 from M2 + gradient cosine-similarity between heads · U-Zeros / U-SelfTrained / U-MultiClass · calibration (ECE, reliability diagrams) · external validation on MIMIC-CXR and PadChest.

---

## Acknowledgments

This project was born from a personal experience with a childhood pneumonia diagnosis — the motivation behind every line of code. Thanks to **Prof. Dr. Gustavo Laureano** for guidance and scientific rigor, and to **Hugo Pessoni** (B.Sc. in AI, UFG) for technical discussions. Institutional support from **SENAI FATESG**, **CEIA/UFG**, and **AKCIT**.

## Author

**Pablo Henrique Miranda Silva** — ML Engineer / AI Researcher
SENAI FATESG · CEIA/UFG · AKCIT — Goiânia, GO, Brazil
[GitHub @Pablo-Henrique-IA](https://github.com/Pablo-Henrique-IA) · [LinkedIn](https://www.linkedin.com/in/pablo-henrique-ia) · pablohmsilva7@gmail.com

## License

<!-- TODO: choose a license (e.g., MIT) and add a LICENSE file.
     Note: NIH ChestX-ray14 and CheXpert have their own usage terms — this license covers the code only. -->
This repository's **code** is released under the MIT License (see `LICENSE`). The datasets are subject to their original licenses.
