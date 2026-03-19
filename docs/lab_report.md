# Lab Report: Implementing a Basic Machine Learning Training Pipeline

**Course:** ML Engineering
**Lab:** Assignment 1
**Submission Date:** 19.03.2026

---

## Introduction

This lab implements a supervised image classification pipeline using a deep learning approach. The chosen problem is **multi-class image classification** — predicting one of 10 object categories from a small color image.

**Dataset: CIFAR-10**

CIFAR-10 is a widely-used benchmark dataset consisting of 60,000 32×32 RGB images across 10 balanced classes:
`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

Each class contains exactly 6,000 images, making the dataset perfectly balanced. The data is distributed as Python pickle files containing raw pixel arrays. In this pipeline, the dataset is downloaded programmatically and split into three sets:
- **Train**: 38,400 images (64%)
- **Validation**: 9,600 images (16%)
- **Test**: 12,000 images (20%)

**Goal:** Train a convolutional neural network that achieves high accuracy on unseen CIFAR-10 images, with the full pipeline demonstrating production-grade practices around configuration management, logging, code quality, and reproducibility.

---

## Pipeline Description

The pipeline is structured as a linear sequence of stages, each implemented as a reusable function:

### Stage 1: Data Download
`download_and_extract(url, save_dir)` downloads the CIFAR-10 tarball from the official Toronto URL (`https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`) and extracts it to `./data/`. It skips re-downloading if the file already exists, making reruns fast.

### Stage 2: Data Ingestion
`load_labels(labels_path)` reads all 6 CIFAR-10 pickle batch files (`data_batch_1` through `data_batch_5` and `test_batch`), saves each of the 60,000 images as a JPEG to `./data/images/<class_name>/`, and returns a unified DataFrame with columns `image_path` and `label` (integer 0–9). This converts the binary CIFAR-10 format into the image-path-based format used by the rest of the pipeline.

`process_data()` then calls `train_test_split()` twice to produce three non-overlapping splits. A fixed `random_state=42` ensures reproducibility across runs.

### Stage 3: Data Augmentation
Two transform pipelines are defined:
- **`train_transform`**: `RandomHorizontalFlip` + `RandomCrop(32, padding=4)` + normalization — reduces overfitting by exposing the model to varied crops and flips during training
- **`eval_transform`**: resize + normalization only — no stochastic augmentation for fair evaluation

CIFAR-10 normalization constants used: `mean=[0.4914, 0.4822, 0.4465]`, `std=[0.2023, 0.1994, 0.2010]`.

### Stage 4: Training Loop
`train_model()` runs for 20 epochs. Each epoch:
1. Sets model to `train()` mode, iterates batches of 128 images, computes cross-entropy loss, runs backpropagation with Adam optimizer (lr=0.001)
2. Sets model to `eval()` mode, computes average validation loss on the val set
3. Saves the model checkpoint whenever validation loss improves (best-model selection)

All epoch-level events (train loss, val loss, checkpoint saves) are recorded with Python's `logging` module.

### Stage 5: Evaluation
`test_model()` loads the best-saved checkpoint and evaluates it on the held-out test set, reporting loss, accuracy, precision, recall, and F1 score via `sklearn.metrics`.

---

## Model Evaluation

**Model: `CifarCNN`**

A custom CNN with three convolutional blocks followed by a fully-connected classifier:

```
Conv2d(3→32, k=3, p=1) → BatchNorm2d(32) → ReLU
Conv2d(32→64, k=3, p=1) → BatchNorm2d(64) → ReLU → MaxPool2d(2)   # 32→16
Conv2d(64→128, k=3, p=1) → BatchNorm2d(128) → ReLU → MaxPool2d(2) # 16→8
Flatten → Dropout(0.5) → Linear(8192→512) → ReLU → Linear(512→10)
```

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epochs | 20 |
| Dropout | 0.5 |

### Training Curve

| Epoch | Train Loss | Val Loss | Checkpoint |
|-------|-----------|----------|------------|
| 1 | 1.4315 | 1.4491 | ✓ saved |
| 2 | 1.2520 | 1.1154 | ✓ saved |
| 3 | 1.1210 | 1.0685 | ✓ saved |
| 4 | 1.1603 | 0.9501 | ✓ saved |
| 5 | 0.9528 | 0.8992 | ✓ saved |
| 6 | 0.8891 | 0.8621 | ✓ saved |
| 7 | 0.9043 | 0.8379 | ✓ saved |
| 8 | 0.9689 | 0.8198 | ✓ saved |
| 9 | 0.9101 | 0.8321 | — |
| 10 | 1.0809 | 0.7999 | ✓ saved |
| 11 | 0.8750 | 0.7781 | ✓ saved |
| 12 | 0.8544 | 0.8047 | — |
| 13 | 0.6363 | 0.7587 | ✓ saved |
| 14 | 0.7811 | 0.8095 | — |
| 15 | 0.7587 | 0.7245 | ✓ saved |
| 16 | 0.7753 | 0.7396 | — |
| 17 | 0.5912 | 0.7189 | ✓ saved |
| 18 | 0.8065 | 0.7183 | ✓ saved |
| **19** | **0.6325** | **0.6826** | **✓ best** |
| 20 | 0.7270 | 0.7315 | — |

The best model was saved at **epoch 19** with validation loss **0.6826**.

### Final Test Results

| Metric | Value |
|--------|-------|
| Test Loss | 0.6857 |
| **Accuracy** | **76.42%** |
| **Precision** (weighted) | **77.07%** |
| **Recall** (weighted) | **76.42%** |
| **F1 Score** (weighted) | **76.42%** |

### Analysis

**Convergence:** The model converged steadily over 20 epochs. Validation loss dropped from 1.45 at epoch 1 to 0.68 at epoch 19 — a 53% reduction — showing consistent learning without significant oscillation.

**Generalization gap:** The gap between the best validation loss (0.6826) and test loss (0.6857) is negligible (~0.003), indicating that Dropout(0.5) combined with data augmentation effectively prevented overfitting. The model generalizes well to unseen data.

**Class balance:** Since CIFAR-10 is perfectly balanced, weighted metrics (precision, recall, F1) are nearly identical to their macro equivalents. The 0.65% spread between precision (77.07%) and recall (76.42%) reflects minor per-class variation typical of visually similar categories (e.g. cat/dog, automobile/truck).

**Performance context:** A 76.42% accuracy on CIFAR-10 from scratch in 20 epochs on CPU is a solid result. The theoretical random-chance baseline is 10%. The original `SimpleNN` template would plateau at ~20–30% (a 2-layer MLP struggles with spatial features). The ~76% figure is consistent with published results for similar 3-block CNN architectures trained without extensive hyperparameter tuning.

**Late-epoch noise:** Epochs 9, 12, 14, and 16 saw val loss spikes despite the overall downward trend, likely due to minibatch stochasticity with a fixed learning rate. The best-model checkpoint mechanism correctly selected epoch 19 over these noisier epochs.

---

## Best Practices

### Configuration Management
All tunable parameters (learning rate, batch size, number of epochs, split sizes, random seed, save path) are stored in a single `config` dictionary in `main()`. No values are hard-coded across function boundaries — every function reads from `config`. This allows experiment sweeps by modifying a single location.

### Logging
`logging.basicConfig(level=logging.INFO)` is configured once at startup. All significant events — batch loading progress, split sizes, per-epoch losses, checkpoint saves, and final metrics — are emitted via `logging.info()` and `logging.error()`. No `print()` statements are used anywhere in the pipeline.

### Code Quality
- All functions and methods have full **type hints** on parameters and return types (verified with `mypy`)
- `ruff`, `black`, and `isort` are listed as dev-dependencies in `pyproject.toml` for linting and formatting
- Code follows consistent style with descriptive variable names and docstrings on all public functions

### Dependency Management
`poetry` is used via `pyproject.toml` to declare all runtime and dev dependencies with pinned or constrained versions. `poetry export` generates a `requirements.txt` for `pip`-based installs, ensuring the environment is reproducible and isolated.

### Version Control
The project is managed with Git. Commits are made at logical checkpoints (initial template, dataset switch, model upgrade, metrics addition) with descriptive messages.

---

## Reflection

**Effectiveness:** `CifarCNN` achieved **76.42% accuracy** on the CIFAR-10 test set after 20 epochs — a strong result for a custom CNN trained from scratch with no pre-training. The training was stable, with no signs of overfitting due to BatchNorm, Dropout, and data augmentation.

**Potential improvements with more time/resources:**

1. **Transfer learning** — Fine-tuning a pretrained ResNet-18 on CIFAR-10 would push accuracy to ~93–95% with minimal additional training
2. **Learning rate scheduling** — `CosineAnnealingLR` or `OneCycleLR` would improve convergence speed and final accuracy (estimated +2–3%)
3. **Longer training with decay** — Training to 100 epochs with a decaying LR would likely push past 80% accuracy
4. **Per-class analysis** — Computing a confusion matrix would reveal which classes the model struggles with (cats vs dogs, automobile vs truck are typical hard pairs in CIFAR-10)
5. **Experiment tracking** — Integrating MLflow or Weights & Biases would allow systematic comparison across hyperparameter configurations and training runs
