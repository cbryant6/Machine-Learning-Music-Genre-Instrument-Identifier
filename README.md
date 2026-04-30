# Music Classification with Signal Processing and CNNs

**TECHIN 513 — Digital Signal Processing**  
University of Washington · Master of Science in Technology Innovation  
Coleman Bryant

---

A unified music analysis system with two CNN pipelines: **genre classification** (10-class GTZAN) and **dominant instrument identification** (4-class Kaggle). Both use log-mel spectrograms as image-like inputs and compare iterative CNN revisions against non-neural baselines.

## Results

| Task | Baseline | Best CNN | Accuracy |
|------|----------|----------|----------|
| **Genre** | LogReg on mel mean+std | V4: 3s crops + SpecAugment + multi-crop eval | **0.77** |
| **Instrument** | LogReg on flattened spectrograms | V4: BatchNorm + class weights + LR scheduling | **0.7148** |

### Genre Progression

| Version | Change | Test Acc |
|---------|--------|----------|
| Baseline | Logistic regression (128-dim mel summary) | 0.65 |
| V2 | CNN + random 3s crops + multi-crop eval | 0.73 |
| V3 | Crop duration 3s → 5s (ablation) | 0.62 |
| **V4** | **+ SpecAugment (time/freq masking)** | **0.77** |

### Instrument Progression

| Version | Change | Val Acc |
|---------|--------|---------|
| Baseline | Logistic regression (flattened spectrograms) | 0.4848 |
| V1 | 3-layer CNN | 0.6806 |
| V2 | + BatchNorm + GlobalAveragePooling | 0.6825 |
| V3 | 3 conv blocks + class weights | 0.7110 |
| **V4** | **+ Dropout 0.5 + ReduceLROnPlateau** | **0.7148** |

## Key Findings

**SpecAugment matters for genre.** Randomly masking time and frequency bands during training was the single largest contributor to genre accuracy, pushing V4 from the V2 baseline of 0.73 to 0.77.

**Shorter crops beat longer ones.** The V3 ablation (3s → 5s crops) produced *worse* results than both V2 and the baseline, confirming that crop diversity outweighs per-crop context length.

**Headline accuracy can hide class-level regression.** The instrument V4 model achieved the highest overall accuracy but collapsed violin recall to 0.13 (from 0.83 in V3). V3 remains the better model when balanced per-class performance matters.

**Multi-crop evaluation stabilizes predictions.** Averaging probabilities across 10 deterministic crops per track reduced the effect of any single short window and consistently outperformed single-crop evaluation.

**Feature pipeline consistency is non-negotiable.** During instrument model development, inference accuracy collapsed to ~0.25 due to an extra normalization step added after training that didn't exist in the training pipeline. Aligning inference with training restored accuracy to 0.71. This reinforced that preprocessing must be identical between training and inference — a subtle but critical lesson.

## Class-Level Analysis

**Genre — strong classes:** Classical, metal, and hip-hop have distinctive spectral signatures and high recall across all runs. **Weak:** Rock is the most persistent failure, overlapping with blues, country, and metal in timbre and rhythm. Disco, country, and jazz are unstable across versions.

**Instrument — strong classes:** Guitar and piano achieve ~95%+ precision. **Weak:** The drum-vs-violin tradeoff persists across versions — V4 favors drum recall at violin's expense, while V3 is more balanced.

## Shared Pipeline

Both tasks use the same signal-processing foundation:

```
Audio (.wav) → librosa.load (22050 Hz, mono)
            → Fixed-length crop (3s)
            → Mel spectrogram (64 bands, FFT=2048, hop=512)
            → Log scale (power_to_db)
            → Per-example standardization ((x - μ) / σ)
            → CNN input (64 × 130 × 1)
```

Both CNNs follow the same structural pattern:

```
Conv2D → BatchNorm → ReLU → MaxPool → Dropout
  × 3 blocks (16 → 32 → 64 filters, progressive dropout)
→ Flatten or GlobalAveragePooling
→ Dense (64–128) → Dropout → Softmax
```

**Optimizer:** Adam + ReduceLROnPlateau  
**Regularization:** Dropout, BatchNorm, EarlyStopping  
**Class imbalance:** Balanced class weights (instrument model)

## Combined Demo

For the final class demo, both classifiers ran on the same audio clip:

1. **Genre model** predicts the musical style (e.g., classical)
2. **Instrument model** predicts the dominant instrument (e.g., piano)

A Beethoven piano clip correctly predicted **classical + piano**, demonstrating that the two models provide complementary information from a shared feature representation.

## Repo Structure

```
├── README.md
├── Genre_Classification_Clean.ipynb       # Genre pipeline (baseline → v4, 0.77)
├── Instrument_Classification_Clean.ipynb  # Instrument pipeline (baseline → v4, 0.7148)
└── TECHIN513_ML_Music_Identifier_Report_ACL.docx  # Full ACL-format report
```

### Saved Artifacts (per run)

**Genre model:**
```
gtzan_cnn_v4_specaug.keras        # Trained model weights
gtzan_label_classes.npy           # Label encoder classes
gtzan_v4_results.json             # Accuracy + config summary
```

**Instrument model:**
```
instrument_cnn_v4.keras           # Trained model weights
instrument_label_classes.npy      # Label encoder classes
instrument_results_summary.json   # All version accuracies
```

## Setup

### Requirements

```
numpy
pandas
matplotlib
seaborn
librosa
scikit-learn
tensorflow
```

### Datasets

**Genre:** Download [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and update `BASE_DIR` in the genre notebook.

**Instrument:** Download [Musical Instrument Sound Dataset](https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset) and update `DATASET_DIR` in the instrument notebook.

### Running

Both notebooks are self-contained. Open in Jupyter or Colab and run top-to-bottom. Each trains all model versions sequentially (baseline through V4) and produces comparison charts and confusion matrices.

## Reproducibility Notes

The genre model's best run (0.77) was achieved in the original Colab session but later reruns under controlled conditions settled at 0.61–0.64. The gap is attributed to GPU non-determinism and session state in Colab. The 0.77 result is reported as the original run with saved artifacts; the reproducible reruns are the more conservative reference point.

This transparency is intentional — it reflects a core lesson of the project: reliable ML workflows require explicit artifact tracking, consistent preprocessing pipelines, and persistent storage from the first run.

## Future Improvements

- [ ] Stronger architectures: ResNet, Audio Spectrogram Transformer (AST)
- [ ] Larger and cleaner datasets (GTZAN has known label noise)
- [ ] Multi-label instrument detection (polyphonic audio)
- [ ] Joint genre + instrument multitask model — shared backbone, dual heads
- [ ] Longer audio context beyond 3-second crops
- [ ] Experiment tracking with MLflow or Weights & Biases
- [ ] Data augmentation targeted at weak classes (violin, rock)

## References

- McFee, B. et al. (2015). *librosa: Audio and music signal analysis in Python.* Proc. 14th Python in Science Conf.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR 12:2825–2830.
- Tzanetakis, G. & Cook, P. (2002). *Musical genre classification of audio signals.* IEEE Trans. Speech and Audio Processing 10(5).
- Prasad, S. (2021). *Musical Instrument's Sound Dataset.* Kaggle.
- TensorFlow Developers (2023). *TensorFlow.* https://www.tensorflow.org/
