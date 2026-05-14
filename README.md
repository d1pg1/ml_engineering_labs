# ML Engineering Labs

Six progressive MLOps labs built on a shared CIFAR-10 classification pipeline.

## Labs Overview

| Lab | Topic | Status |
|-----|-------|--------|
| [Lab 01](notebooks/ml_engineering_lab_01.ipynb) | Image Classification Pipeline (PyTorch, CIFAR-10) | ✅ Complete |
| [Lab 02](notebooks/ml_engineering_lab_02.ipynb) | Automating Dataset Extension (config-driven batch selection) | — |
| [Lab 03](notebooks/ml_engineering_lab_03.ipynb) | DVC Pipeline Automation (data versioning, `dvc repro`) | — |
| [Lab 04](notebooks/ml_engineering_lab_04.ipynb) | MLflow Experiment Tracking & Artifact Management | — |
| [Lab 05](notebooks/ml_engineering_lab_05.ipynb) | Weights & Biases Experiment Tracking | — |
| [Lab 06](notebooks/ml_engineering_lab_06.ipynb) | Streamlit Interactive Model Analysis Dashboard | — |

## Lab 01 Results

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
├── src/                              # Shared ML source (extracted from Lab 01)
│   ├── data/
│   │   ├── download.py               # download_and_extract()
│   │   └── dataset.py                # ImageDataset, loaders, splits
│   ├── models/
│   │   └── cnn.py                    # CifarCNN
│   └── training/
│       ├── trainer.py                # train_model()
│       └── evaluate.py               # test_model()
│
├── configs/
│   └── base.yaml                     # Base training config (all labs)
│
├── scripts/                          # DVC pipeline entry points (Lab 03+)
│   ├── download_data.py
│   ├── train.py
│   └── evaluate.py
│
├── dashboard/                        # Streamlit app (Lab 06)
│   ├── app.py
│   └── utils/
│
├── notebooks/                        # One notebook per lab
│   └── ml_engineering_lab_0*.ipynb
│
├── outputs/
│   └── best_model.pth                # Lab 01 best checkpoint
│
├── docs/
│   └── lab_report_0*.md              # Lab reports
│
├── tasks/                            # Assignment PDFs
├── params.yaml                       # DVC parameter file (Lab 03+)
├── pyproject.toml                    # Poetry dependencies
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS

pip install -r requirements.txt
```

## Running a Lab Notebook

```bash
.venv/bin/jupyter notebook notebooks/ml_engineering_lab_01.ipynb
```

## Running the Streamlit Dashboard (Lab 06)

```bash
streamlit run dashboard/app.py
```

## Running the DVC Pipeline (Lab 03+)

```bash
dvc repro
```

## Dataset

**CIFAR-10** — 60,000 32×32 RGB images across 10 classes.
Downloaded automatically from `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`.

## Dependencies

Key libraries: `torch`, `torchvision`, `scikit-learn`, `pandas`, `Pillow`, `numpy`.  
Dev tools: `mypy`, `ruff`, `black`, `isort`.

See [pyproject.toml](pyproject.toml) for the full specification or [requirements.txt](requirements.txt) for pinned versions.
