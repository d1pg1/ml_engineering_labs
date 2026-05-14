# ML Engineering Labs

Six progressive MLOps labs built on a shared CIFAR-10 classification pipeline.

## Labs Overview

| Lab | Topic | Notebook / Entry point |
|-----|-------|----------------------|
| [Lab 01](docs/lab_report_01.md) | Image Classification Pipeline (PyTorch, CIFAR-10) | [notebook](notebooks/ml_engineering_lab_01.ipynb) |
| [Lab 02](docs/lab_report_02.md) | Config-Driven Dataset Extension & Multi-Run Comparison | [notebook](notebooks/ml_engineering_lab_02.ipynb) |
| [Lab 03](docs/lab_report_03.md) | DVC Pipeline Automation (data versioning, `dvc repro`) | [notebook](notebooks/ml_engineering_lab_03.ipynb) |
| [Lab 04](docs/lab_report_04.md) | MLflow Experiment Tracking & Artifact Management | [notebook](notebooks/ml_engineering_lab_04.ipynb) |
| [Lab 05](docs/lab_report_05.md) | Weights & Biases Experiment Tracking | [notebook](notebooks/ml_engineering_lab_05.ipynb) |
| [Lab 06](docs/lab_report_06.md) | Streamlit Interactive Model Analysis Dashboard | [dashboard/app.py](dashboard/app.py) |

## Project Structure

```
.
├── src/                              # Shared ML source
│   ├── data/
│   │   └── dataset.py                # ImageDataset, loaders, splits
│   ├── models/
│   │   └── cnn.py                    # CifarCNN (3-block CNN, BatchNorm + Dropout)
│   └── training/
│       ├── trainer.py                # train_model()
│       ├── evaluate.py               # test_model()
│       ├── mlflow_trainer.py         # MLflow-instrumented trainer (Lab 04)
│       └── wandb_trainer.py          # W&B-instrumented trainer (Lab 05)
│
├── configs/
│   ├── base.yaml                     # Base training config
│   ├── lab02.yaml                    # Lab 02 experiment variants
│   ├── lab04.yaml                    # Lab 04 sweep config
│   └── lab05.yaml                    # Lab 05 sweep config
│
├── scripts/                          # DVC pipeline entry points
│   ├── download_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── setup_mlflow.py              # Bootstrap MLflow runs from existing checkpoints
│
├── dashboard/                        # Streamlit app (Lab 06)
│   ├── app.py                        # Entry point (3 tabs)
│   ├── config.yaml                   # Dashboard paths and settings
│   └── utils/
│       ├── config.py
│       ├── data_utils.py
│       ├── mlflow_utils.py
│       ├── model_utils.py            # Inference + Grad-CAM
│       └── viz_utils.py
│
├── notebooks/                        # One notebook per lab (Labs 01–05)
│
├── outputs/                          # Trained checkpoints
│   ├── lab04_main.pth
│   ├── lab04_sweep_lr*.pth
│   ├── lab05_main.pth
│   ├── lab05_sweep_lr*.pth
│   └── metrics.json
│
├── data/                             # CIFAR-10 raw batches + extracted images
│   ├── cifar-10-batches-py/
│   └── images/                       # Per-class PNG exports
│
├── docs/
│   ├── lab_report_0*.md              # Lab reports
│   └── screenshots/                  # Dashboard screenshots (Lab 06)
│
├── tasks/                            # Assignment PDFs
├── dvc.yaml                          # DVC pipeline definition (Lab 03+)
├── dvc.lock
├── params.yaml                       # DVC parameter file
├── configs/base.yaml                 # Base training hyperparameters
└── pyproject.toml                    # uv-managed dependencies
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
uv venv --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[dev]"
```

For CPU-only PyTorch, omit the `--index-url` flag (defaults to CPU wheels).

## Running a Lab Notebook

```bash
STREAMLIT_SERVER_HEADLESS=true .venv/bin/jupyter notebook notebooks/ml_engineering_lab_01.ipynb
```

## Running the DVC Pipeline (Lab 03+)

```bash
dvc repro
```

## Running the Streamlit Dashboard (Lab 06)

```bash
# Bootstrap MLflow tracking data (run once)
.venv/bin/python scripts/setup_mlflow.py

# Launch dashboard
STREAMLIT_SERVER_HEADLESS=true .venv/bin/python -m streamlit run dashboard/app.py --server.port 8501
```

The dashboard exposes three tabs: **Dataset Exploration**, **Error Analysis** (batch inference + confusion matrix + Grad-CAM misclassified browser), and **Prediction & Explainability** (per-sample Grad-CAM overlay).

## Dataset

**CIFAR-10** — 60,000 32×32 RGB images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Split: 38,400 train / 9,600 val / 12,000 test.

Downloaded automatically via `scripts/download_data.py` or the Lab 01 notebook.

## Dependencies

| Group | Key libraries |
|-------|--------------|
| Core ML | `torch`, `torchvision`, `scikit-learn`, `numpy`, `pandas`, `Pillow` |
| Experiment tracking | `mlflow`, `wandb` |
| Dashboard | `streamlit`, `plotly`, `lime` |
| Pipeline | `dvc` |
| Dev | `ruff`, `black`, `mypy`, `ipykernel`, `nbconvert` |

See [pyproject.toml](pyproject.toml) for the full specification.
