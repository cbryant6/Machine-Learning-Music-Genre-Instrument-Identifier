# 🎵 Audio Classification with CNNs: Genre & Instrument Recognition

Two deep learning pipelines for music understanding — both trained on log-mel spectrograms using convolutional neural networks.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Module 1: Music Genre Classification (GTZAN)](#module-1-music-genre-classification-gtzan)
- [Module 2: Instrument Identification](#module-2-instrument-identification)
- [Shared Architecture & Design Decisions](#shared-architecture--design-decisions)
- [Results Summary](#results-summary)
- [Reproducibility Notes](#reproducibility-notes)
- [Saved Artifacts](#saved-artifacts)
- [Future Improvements](#future-improvements)

---

## Project Overview

This repository contains two CNN-based audio classifiers built as part of a deep learning exploration into music understanding:

| Module | Task | Classes | Best Test Accuracy |
|---|---|---|---|
| Genre Classification | Predict music genre | 10 (blues, classical, rock, ...) | .77 |
| Instrument Identification | Predict instrument | 4 (drum, guitar, piano, violin) | ~0.71 |

Both models share the same core approach:
- Audio → log-mel spectrogram (64 mel bands, 22,050 Hz, 3-second clips)
- CNN with BatchNorm, Dropout, Global/Max Pooling
- Adam optimizer with `ReduceLROnPlateau` and `EarlyStopping`
- Artifacts saved to Google Drive per run

---

## Module 1: Music Genre Classification (GTZAN)

### Dataset

- **Source:** GTZAN Dataset
- **Classes (10):** blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- **~100 tracks/genre** (after filtering one corrupted file)
- **Split:** Train: 799 tracks | Val: 100 | Test: 100
- Each track split into 3-second crops for training

### Key Techniques

**Multi-Crop Training & Evaluation**
- Each track yields 4–6 overlapping crops during training
- At test time, predictions are averaged across crops for stability

**SpecAugment (Data Augmentation)**
- Time masking and frequency masking applied during training
- Controlled ON/OFF ablations to measure impact

### Ablation Experiments

All runs used `SEED = 513` for reproducibility.

| Run | Setting | Test Accuracy |
|---|---|---|
| Baseline | Simple CNN | 0.65 |
| CNN v2 | Improved architecture | 0.73 |
| CNN v3 | Modified pipeline | 0.62 |
| CNN v4 (early run) | Best recorded | **0.77** |
| CNN v4 recreated | SpecAug ON, crops=4 | 0.63 |
| CNN v4 ablation | crops=6 | 0.61 |
| CNN v4 ablation | SpecAug OFF | 0.64 |

### Class-Level Performance

**Strong classes:** Classical, Metal, Hip-hop — distinct spectral signatures, high recall

**Weak classes:** Rock (hardest), Country, Disco, Jazz — overlapping timbre and rhythm patterns

Misclassification patterns are consistent with known GTZAN dataset ambiguity and the limited 3-second context window.

---

## Module 2: Instrument Identification

### Dataset

- **Source:** Kaggle Musical Instruments Sound Dataset
- **Classes (4):** Drum, Guitar, Piano, Violin
- **Format:** `.wav`, 3 seconds, 22,050 Hz
- **Split:** Train / Validation / Test

### Feature Extraction Pipeline

```python
librosa.load(path, sr=22050)
# Trim/pad to fixed 3-second length
mel = librosa.feature.melspectrogram(y, sr=22050, n_mels=64, n_fft=2048, hop_length=512)
log_mel = librosa.power_to_db(mel, ref=np.max)
```

> **Note:** No per-sample normalization is applied at inference — this must match training exactly.

### Key Fix During Development

Inference accuracy collapsed to ~0.25 due to a feature pipeline mismatch (extra normalization added post-training). After aligning inference with training:

```python
def featurize_path(path):
    return wav_to_logmel(path)  # No additional normalization
```

Test accuracy restored to ~0.61.

### Class-Level Performance

**Strong:** Drum, Piano — highly distinct spectral and rhythmic signatures

**Moderate:** Guitar

**Weakest:** Violin — most challenging, benefits most from augmentation

### Inference

```python
model = tf.keras.models.load_model("instrument_cnn_v4.keras")
class_names = np.load("instrument_label_classes.npy", allow_pickle=True)

def predict_instrument(path):
    feat = wav_to_logmel(path)
    x = feat[np.newaxis, ..., np.newaxis]
    probs = model.predict(x, verbose=0)[0]
    print("Prediction:", class_names[np.argmax(probs)])
    print("Confidence:", round(float(np.max(probs)), 3))
```

---

## Shared Architecture & Design Decisions

Both models use the same CNN backbone:

```
Input (log-mel spectrogram, shape: [T, 64, 1])
→ Conv2D → BatchNorm → ReLU → MaxPool
→ Conv2D → BatchNorm → ReLU → MaxPool
→ Conv2D → BatchNorm → ReLU → MaxPool/GlobalAvgPool
→ Flatten / GAP
→ Dense (64 units) → Dropout
→ Dense (N classes, Softmax)
```

- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam + `ReduceLROnPlateau`
- **Regularization:** Dropout, BatchNorm, early stopping
- **Class imbalance:** Handled via class weights (instrument model)

---

## Results Summary

| Model | Task | Val Accuracy | Test Accuracy |
|---|---|---|---|
| CNN v4 (genre) | 10-class genre | — | ~0.63–0.64 |
| CNN v4 (instrument) | 4-class instrument | ~0.71 | ~0.61 |

Both models are limited by:
- Small dataset size (GTZAN: ~1000 tracks; instrument dataset similarly small)
- Short 3-second audio windows losing long-range musical context
- Genre/instrument overlap in spectral features

---

## Reproducibility Notes

A key lesson from this project:

The genre model's best run (0.77 accuracy) could **not** be reproduced under controlled reconstruction. Root causes:
- Google Drive was not mounted before training — artifacts were lost
- Model weights, logs, and dataset splits were not preserved

Later reruns with identical seeds produced stable but lower results (~0.63–0.64).

The instrument model surfaced a related issue: a subtle feature pipeline mismatch between training and inference caused accuracy to drop to chance. Aligning the pipelines restored performance.

**Takeaway:** Reliable ML workflows require explicit artifact tracking, consistent preprocessing pipelines, and persistent storage from the first run.

---

## Saved Artifacts

### Genre Model

```
gtzan_ml/
├── gtzan_cnn_v4_no_specaug.keras
├── gtzan_label_classes.npy
├── cnn_v4_no_specaug_results.txt
├── cnn_v4_no_specaug_confusion_counts.png
└── cnn_v4_no_specaug_confusion_normalized.png
```

### Instrument Model

```
instrument_ml/
├── instrument_cnn_v4.keras
├── instrument_label_classes.npy
├── instrument_test_results.json
├── instrument_test_predictions.csv
└── test_meta.csv
```

Each run saves: trained model, label encoder, confusion matrices, classification report, and experiment config.

---

## Future Improvements

- [ ] Stronger architectures: ResNet, Audio Spectrogram Transformer (AST)
- [ ] Larger and cleaner datasets (GTZAN has known label noise)
- [ ] Cross-dataset validation to test generalization
- [ ] Multi-label instrument detection (polyphonic audio)
- [ ] **Joint genre + instrument multitask model** — shared backbone, dual heads
- [ ] Longer audio context (beyond 3-second crops)
- [ ] Experiment tracking with MLflow or Weights & Biases
- [ ] Data augmentation targeted at weak classes (violin, rock)
