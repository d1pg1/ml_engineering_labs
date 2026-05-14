# Lab Report: Automating Dataset Extension

**Course:** ML Operations
**Lab:** Assignment 2

---

## Introduction

This lab extends the Lab 01 CIFAR-10 image classification pipeline by introducing **configuration-driven batch selection** — a system for dynamically controlling which portions of the training data are used in each experiment.

**Problem domain:** The core challenge in data management is enabling flexible experimentation without sacrificing reproducibility. When researchers explore how different quantities or compositions of training data affect model performance, they need a system where the test set is always held constant (for fair comparison) while the train/validation composition changes without touching code.

**Dataset: CIFAR-10**

CIFAR-10 consists of 60,000 32×32 RGB images across 10 balanced classes. In this lab, the full dataset is partitioned as follows:

- **Static test set**: 12,000 images (20%), created once with a fixed random seed — never changes across experiments
- **Train+val pool**: 48,000 images (80%), divided into 5 equal batches of ~9,600 images each
- Different experiments draw different numbers of batches from this pool for training, while validation always uses the same single held-out batch

**Goals:**
1. Implement a configuration-driven batch selection system with `configs/lab02.yaml`
2. Demonstrate how training set size (controlled by batch count) affects model performance
3. Maintain a static test set so that cross-experiment metric comparisons are valid

---

## Pipeline Description

### Configuration Management

All parameters are controlled via `configs/lab02.yaml`:

```yaml
data:
  test_size: 0.2
  val_size: 0.2
  random_state: 42
  cifar_url: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
  data_dir: "data"
  n_classes: 10

batches:
  total: 5        # divide train+val pool into this many equal batches
  train: 3        # default number of batches used for training
  val: 1          # number of batches used for validation
  seed: 42        # seed for reproducible batch assignment

training:
  lr: 0.001
  batch_size: 128
  num_epochs: 10
  num_workers: 2

output:
  save_path: "outputs/lab02_model.pth"
```

This YAML is loaded at runtime via `yaml.safe_load()`. No hyperparameters are hard-coded in the notebook or source files.

### Data Combination Script — `src/data/batch_dataset.py`

The new `batch_dataset.py` module implements three functions:

**`divide_into_batches(df, n_batches, random_state)`**
Shuffles the DataFrame once with a fixed seed, then splits it into `n_batches` equal-sized chunks using integer index slicing (`df.iloc[range(i*n//k, (i+1)*n//k)]`). The fixed shuffle seed guarantees that the same images always land in the same batches across runs.

**`select_batches(batches, batch_ids)`**
Concatenates the specified batch DataFrames by index into a single DataFrame. This is what enables "Experiment A uses batches [0]" vs "Experiment B uses batches [0,1,2]".

**`create_batch_splits(df, config)`**
Orchestrates the full pipeline:
1. Creates the **static test set** by calling `train_test_split(df, test_size=0.2, random_state=42)` — this is always the same 12,000 images
2. Calls `divide_into_batches()` to split the remaining 48,000 images into 5 batches
3. Assigns training batches (indices 0 to n_train-1) and the validation batch (index n_train)
4. Returns `(train_df, val_df, test_df)`

### Dynamic Train/Val Sets — Static Test Set

The key invariant is:

```
test_df  ← created once with random_state=42, never modified
batches  ← pool divided with batch.seed=42, indices always consistent

Experiment A: train=batches[0],        val=batches[4]
Experiment B: train=batches[0,1,2],    val=batches[4]
Experiment C: train=batches[0,1,2,3],  val=batches[4]
```

Because the test set is carved out before any batch division, and the batch shuffle uses a fixed seed, every experiment evaluates on exactly the same 12,000 test images.

### Training Loop

Each experiment reuses the existing `src/training/trainer.py` and `src/training/evaluate.py` from Lab 01:
- `train_model()`: 10 epochs, Adam optimizer (lr=0.001), best-checkpoint selection by validation loss
- `test_model()`: evaluates the best checkpoint on the static test set, returns accuracy, precision, recall, F1

---

## Model Evaluation

**Model: `CifarCNN`** (same architecture as Lab 01)

```
Conv2d(3→32, k=3, p=1) → BatchNorm2d(32) → ReLU
Conv2d(32→64, k=3, p=1) → BatchNorm2d(64) → ReLU → MaxPool2d(2)
Conv2d(64→128, k=3, p=1) → BatchNorm2d(128) → ReLU → MaxPool2d(2)
Flatten → Dropout(0.5) → Linear(8192→512) → ReLU → Linear(512→10)
```

### Experiment Configurations

| Experiment | Train Batches | Train Samples | Val Samples | Test Samples |
|-----------|:---:|---:|---:|---:|
| A | 1 of 5 | 9,600 | 9,600 | 12,000 |
| B | 3 of 5 | 28,800 | 9,600 | 12,000 |
| C | 4 of 5 | 38,400 | 9,600 | 12,000 |

### Final Test Metrics (Static Test Set)

| Experiment | Train Samples | Test Loss | Accuracy | Precision | Recall | F1 |
|-----------|---:|---:|---:|---:|---:|---:|
| A (1 batch) | 9,600 | 1.0386 | 63.21% | 63.07% | 63.21% | 62.88% |
| B (3 batches) | 28,800 | 0.8885 | 68.44% | 69.02% | 68.44% | 67.86% |
| **C (4 batches)** | **38,400** | **0.8260** | **71.03%** | **71.77%** | **71.03%** | **70.67%** |

### Analysis

**Effect of training set size:** Accuracy increased monotonically with the number of training batches, confirming the expected scaling behavior:

- A → B (+3 batches, 3× more data): **+5.23% accuracy** (63.21% → 68.44%)
- B → C (+1 batch, 1.33× more data): **+2.59% accuracy** (68.44% → 71.03%)
- A → C (+3 batches, 4× more data): **+7.82% accuracy** total

This follows the well-established data scaling law: more labeled data leads to better generalization on held-out test sets, as long as the model has sufficient capacity.

**Diminishing returns:** The step from 1→3 batches (3× more data) yielded +5.23% accuracy, while 3→4 batches (1.33× more data) added only +2.59%. This diminishing-returns pattern is typical — the marginal value of additional training examples decreases as the model has already learned the core feature distributions. The largest gains come from the early data additions.

**Static test set validity:** Because all three experiments evaluate on exactly the same 12,000 test images (fixed by `random_state=42`), the metric differences are attributable solely to the change in training data composition. This is the primary advantage of the batch selection system: separating "what we train on" from "what we evaluate on."

**Validation vs test alignment:** Test loss closely tracks validation loss in all three experiments, indicating that the best-checkpoint selection (based on val loss) works correctly. In Experiment C, the best validation loss reached 0.8132 at epoch 10, and the corresponding test loss was 0.8260 — a small gap of 0.013, which shows good generalization.

**10 epochs vs 20 epochs:** Lab 01 trained for 20 epochs and achieved 76.42% accuracy with 38,400 training samples. Experiment C (same training set, 10 epochs) reached 71.03%, showing that the extra 10 epochs in Lab 01 contributed ~5% additional accuracy. With 10 epochs, the model had not yet fully converged, particularly for the smaller training sets.

---

## Best Practices

### Configuration Management
All tunable parameters (batch counts, learning rate, epochs, split sizes, random seeds, save path) live exclusively in `configs/lab02.yaml`. The notebook and source modules read from this file — no values are hard-coded in function arguments. Switching from 3 training batches to 4 requires changing one number in the YAML, with no code modification.

### Logging
Python's `logging` module is used throughout all source files (`src/data/batch_dataset.py`, `src/data/dataset.py`, `src/training/trainer.py`, `src/training/evaluate.py`). Key events logged:
- Data download start/completion
- Image batch loading progress
- Batch division results (number of batches, samples per batch)
- Per-epoch train and validation loss
- Checkpoint saves
- Final test metrics

No `print()` statements are used in any source module.

### Code Quality
- **Type hints**: Full type annotations on all functions and return types (`pd.DataFrame`, `List[pd.DataFrame]`, `Dict[str, Any]`, etc.)
- **Docstrings**: All public functions have argument/return documentation matching the style in `dataset.py`
- **Linting**: `ruff`, `black`, and `isort` declared as dev-dependencies in `pyproject.toml`
- **Static analysis**: `mypy` declared as a dev-dependency for type checking

### Dependency Management
Poetry manages all runtime and development dependencies via `pyproject.toml`, with `requirements.txt` generated for `pip`-based installs. The `.venv` virtual environment ensures isolation from the system Python.

### Version Control
The project uses Git with meaningful, task-scoped commit messages. The new `src/data/batch_dataset.py` module and `configs/lab02.yaml` are committed as a logical unit separate from the notebook implementation.

---

## Reflection

**Effectiveness:** The batch selection system achieved its primary goal: a clean, config-driven way to control training data composition while keeping the evaluation baseline constant. Experiment C (4 batches, 38,400 samples, 10 epochs) reached **71.03% accuracy**, Experiment B (3 batches, 28,800 samples) achieved **68.44%**, and Experiment A (1 batch, 9,600 samples) achieved **63.21%** — a clear, monotonic progression that confirms the expected data scaling relationship.

**Design tradeoffs:**
- *Simple batch indexing* (first N batches for train, next M for val) is reproducible and easy to reason about, but doesn't guarantee class-balanced batches per-experiment. A stratified batch split would be more robust for imbalanced datasets.
- *Fixed val batch* (batch 4 in all experiments) means val and train share no samples, but the val composition doesn't change between experiments. This is intentional — it isolates the independent variable (training data size) from confounders.

**Potential improvements with more time:**
1. **Stratified batch creation** — ensure each batch has the same class distribution, avoiding accidentally easier or harder train/val splits
2. **Batch composition variation** — beyond "how many batches" (quantity), investigate "which batches" (composition) — e.g., what happens if we pick 3 batches that contain mostly common classes vs. rare classes
3. **Learning rate scheduling** — `CosineAnnealingLR` across the 10-epoch runs would likely improve convergence and increase the gap between experiments
4. **More epochs** — 10 epochs is sufficient to show the trend, but 20 epochs (as in Lab 01) would push absolute accuracy higher and show clearer separation between configurations
5. **DVC integration** — parameterizing batch counts via `dvc params` would allow `dvc exp run` to automatically sweep configurations and compare results in a tracked experiment grid
