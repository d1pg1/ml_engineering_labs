# ML Engineering — Lab 01: Image Classification Pipeline

Supervised image classification pipeline built with PyTorch on the **CIFAR-10** dataset.
Implements data download, ingestion, augmentation, training, and evaluation as a reproducible
end-to-end pipeline following ML engineering best practices.

## Results

| Metric | Value |
|--------|-------|
| Accuracy | **76.42%** |
| Precision (weighted) | 77.07% |
| Recall (weighted) | 76.42% |
| F1 Score (weighted) | 76.42% |
| Test Loss | 0.6857 |

Model: `CifarCNN` (3-block CNN with BatchNorm + Dropout), trained for 20 epochs on CPU.

## Project Structure

```
.
├── notebooks/
│   └── ml_engineering_lab_01.ipynb   # Main pipeline notebook
├── outputs/
│   └── best_model.pth                # Best checkpoint (saved during training)
├── data/                             # Downloaded dataset (auto-created, gitignored)
│   ├── cifar-10-batches-py/          # Raw CIFAR-10 pickle files
│   └── images/                       # Extracted JPEGs organized by class
├── docs/
│   └── lab_report.md                 # Full lab report with results analysis
├── .venv/                            # Local virtual environment (gitignored)
├── pyproject.toml                    # Project metadata and dependencies (poetry)
├── requirements.txt                  # Pinned dependencies for pip
├── .gitignore
└── README.md
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Notebook

```bash
# From the repo root
.venv/bin/jupyter notebook notebooks/ml_engineering_lab_01.ipynb
```

Or execute non-interactively:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute \
    notebooks/ml_engineering_lab_01.ipynb \
    --output notebooks/ml_engineering_lab_01_executed.ipynb \
    --ExecutePreprocessor.timeout=3600
```

The pipeline will:
1. Download CIFAR-10 (~170 MB) into `data/` — skipped if already present
2. Extract images to `data/images/<class>/`
3. Split into train (64%) / val (16%) / test (20%)
4. Train `CifarCNN` for 20 epochs, saving the best checkpoint to `outputs/best_model.pth`
5. Evaluate and log accuracy, precision, recall, F1

## Dataset

**CIFAR-10** — 60,000 32×32 RGB images across 10 classes:
`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

Downloaded automatically from `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`.

## Dependencies

Key libraries: `torch`, `torchvision`, `scikit-learn`, `pandas`, `Pillow`, `numpy`, `scipy`.

Dev tools: `mypy`, `ruff`, `black`, `isort`.

See [pyproject.toml](pyproject.toml) for full specification or [requirements.txt](requirements.txt) for pinned versions.
