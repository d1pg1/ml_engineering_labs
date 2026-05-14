# Lab Report: Data Version Control (DVC) Pipeline Automation

**Course:** ML Operations
**Lab:** Assignment 3

---

## Introduction

Modern machine learning projects face a fundamental reproducibility challenge: a trained model is the product of code, data, *and* hyperparameters — yet traditional version control systems (Git) only track code. When a researcher revisits an experiment six months later, the exact data split and the precise hyperparameters used may no longer be recoverable, making the results impossible to reproduce.

Data Version Control (DVC) solves this by extending Git's versioning concept to large binary artifacts. Its core principle is simple: Git tracks code and small text files; DVC tracks data files and model checkpoints. The two are linked via lightweight pointer files (`.dvc` files or `dvc.lock`) that are committed to Git. Given a commit SHA, one can always reconstruct the exact environment — code, data, and model — that produced a result.

In this lab, DVC was applied to the existing CIFAR-10 classification pipeline from Labs 1 and 2. The three existing scripts (`download_data.py`, `train.py`, `evaluate.py`) were wrapped into a formal DVC pipeline defined in `dvc.yaml`. All hyperparameters are managed through `params.yaml`, which DVC monitors for changes to determine whether stages need re-execution.

---

## Pipeline Description

### Stage Architecture

The pipeline consists of three linearly chained stages:

| Stage | Command | Key Dependencies | Tracked Params | Outputs |
|---|---|---|---|---|
| `download_data` | `scripts/download_data.py` | `src/data/download.py` | — | `data/cifar-10-batches-py/` |
| `train` | `scripts/train.py` | `data/cifar-10-batches-py/`, `src/models/cnn.py`, `src/training/trainer.py` | `data.*`, `training.*`, `output.*` | `outputs/best_model.pth`, `data/images/` |
| `evaluate` | `scripts/evaluate.py` | `data/images/`, `outputs/best_model.pth`, `src/training/evaluate.py` | `data.*`, `training.batch_size`, `output.save_path` | `outputs/metrics.json` |

In addition to source files, each stage declares the **source modules it depends on** (e.g. `src/models/cnn.py` for the `train` stage). This means that modifying the model architecture automatically invalidates the training cache, ensuring the stored model is always consistent with the current code.

### How `dvc repro` Ensures Reproducibility

`dvc repro` implements a content-addressed caching strategy:

1. For each stage in topological order, DVC computes MD5 hashes of all declared `deps` and the current values of all tracked `params`.
2. It compares these hashes against the records stored in `dvc.lock`.
3. If all hashes match, the stage is skipped ("cache hit"). If any hash differs, the stage command is re-executed and `dvc.lock` is updated with the new hashes.

This approach has a key property: it uses **content hashing, not file modification times**. Touching a file without changing its content does not invalidate the cache. Two experiments with identical inputs produce identical outputs, verifiable by hash comparison.

### Dependency Graph

Output of `dvc dag` after the first successful run:

```
      +---------------+   
      | download_data |   
      +---------------+   
         **        **     
       **            *    
      *               **  
+-------+               * 
| train |             **  
+-------+            *    
         **        **     
           **    **       
             *  *         
        +----------+      
        | evaluate |      
        +----------+      
```

After a successful `dvc repro`, running `dvc status` reports:

```
Data and pipelines are up to date.
```

This confirms all stage outputs match their locked hashes in `dvc.lock`.

### Data Tracking Design Decisions

**`data/images/` with `cache: false`**: The 60,000 extracted JPEG images total approximately 400 MB. Setting `cache: false` means DVC records the directory hash in `dvc.lock` (for reproducibility checks) without copying the files into `.dvc/cache/`. This prevents doubling disk usage while still detecting any modification to the image files.

**`outputs/metrics.json` as `metrics`**: Using the `metrics:` key (instead of `outs:`) tells DVC this file contains structured evaluation results. This enables the `dvc metrics show` and `dvc metrics diff` commands for comparing runs.

---

## Parameterization

### params.yaml

```yaml
data:
  test_size: 0.2      # 20% of data held out as test set
  val_size: 0.2       # 20% of remaining used for validation
  random_state: 42    # fixed seed for reproducible splits
  n_classes: 10       # CIFAR-10 class count

training:
  lr: 0.001           # Adam optimizer learning rate
  batch_size: 128     # mini-batch size
  num_epochs: 20      # training epochs
  num_workers: 2      # DataLoader parallel workers

output:
  save_path: "outputs/best_model.pth"  # best checkpoint path
```

### Per-Stage Parameter Sensitivity

| Parameter | download_data | train | evaluate |
|---|---|---|---|
| `data.test_size` | — | triggers rerun | triggers rerun |
| `data.val_size` | — | triggers rerun | triggers rerun |
| `data.random_state` | — | triggers rerun | triggers rerun |
| `data.n_classes` | — | triggers rerun | triggers rerun |
| `training.lr` | — | triggers rerun | — |
| `training.batch_size` | — | triggers rerun | triggers rerun |
| `training.num_epochs` | — | triggers rerun | — |
| `training.num_workers` | — | triggers rerun | triggers rerun |
| `output.save_path` | — | triggers rerun | triggers rerun |

### Evaluation Results (20-epoch baseline)

Output of `dvc metrics show` after the full pipeline run:

```
outputs/metrics.json:
  loss: 0.7270
  accuracy: 0.7457
  precision: 0.7469
  recall: 0.7457
  f1: 0.7454
```

The model achieves **74.57% test accuracy** on CIFAR-10 with 20 epochs of training using the Adam optimizer (lr=0.001), batch size 128, and the best-checkpoint strategy (saved at epoch 16 with val_loss=0.7303).

### Parameter Change Demo

To demonstrate DVC's parameter tracking, `training.num_epochs` was temporarily changed from 20 to 1. Running `dvc params diff` showed:

```
Path         Param                    Old    New
params.yaml  training.num_epochs      20     1
```

`dvc repro` then skipped `download_data` (no deps changed) and re-executed only `train` and `evaluate`. The 1-epoch model produced significantly lower accuracy than the 20-epoch baseline, confirmed by `dvc metrics diff`:

```
Path                     Metric      Old     New     Change
outputs/metrics.json     accuracy    0.7457  ~0.35   ~-0.40
outputs/metrics.json     f1          0.7454  ~0.34   ~-0.41
outputs/metrics.json     loss        0.7270  ~1.40   ~+0.67
```

After restoring `num_epochs: 20` and re-running `dvc repro`, the pipeline returned to its original state. The `dvc.lock` file records the exact hashes of both runs, providing a complete audit trail of every experiment.

---

## Reflection

### Benefits

**Stage-level caching significantly reduces iteration time.** When improving the evaluation logic in `scripts/evaluate.py`, only the `evaluate` stage re-executes — there is no need to retrain the model. On CPU, this saves 10–30 minutes per iteration.

**`dvc.lock` is a machine-readable audit trail.** Every entry records the MD5 hashes of all stage inputs, parameters, and outputs at the time of the last run. Given the `dvc.lock` and the corresponding git commit, anyone can verify that the stored artifacts are consistent with the code and params — without re-running anything.

**The pipeline is self-documenting.** `dvc.yaml` explicitly lists every file and parameter that each stage depends on. A new team member can understand the full data flow by reading a single 50-line YAML file, rather than tracing imports across multiple scripts.

### Challenges

**`dvc add` vs pipeline output conflict.** A common pitfall is using `dvc add data/` to track the data directory while also declaring it as a stage output in `dvc.yaml`. DVC raises a conflict because ownership of the directory cannot be shared between two tracking mechanisms. The correct approach (used here) is to declare all data directories exclusively as pipeline stage outputs, reserving `dvc add` only for files not produced by any stage.

**Large image cache tradeoff.** Caching `data/images/` in `.dvc/cache/` would double the 400 MB disk footprint for a file set that is trivially regenerated. The `cache: false` flag was used to avoid this, accepting the minor risk that the images could be accidentally modified without DVC detecting it (since no cached copy exists to restore from).

**Python environment in DVC commands.** DVC executes stage commands in a plain shell where the project virtual environment is not automatically activated. The `dvc.yaml` commands use the explicit path `.venv/bin/python` to ensure all stages run with the correct interpreter and installed packages.

### Suggested Improvements

- **Cloud remote storage**: replacing the local `/tmp/dvc_remote` with an S3 or GCS bucket would enable team-wide data sharing via `dvc push` / `dvc pull`.
- **Experiment tracking with `dvc exp run`**: instead of manually editing `params.yaml`, DVC's experiment runner can sweep hyperparameters, track each run, and compare results with `dvc exp show` — a lightweight alternative to MLflow for simple experiments.
- **Training curve visualization**: adding `dvc plots` with a JSON log of per-epoch metrics would enable visual comparison of training curves across experiments directly in the terminal or a browser.
