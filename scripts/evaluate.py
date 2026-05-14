"""DVC stage: evaluate the trained model on the test set."""
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_data_loader, eval_transform, process_data
from src.models.cnn import CifarCNN
from src.training.evaluate import test_model

logging.basicConfig(level=logging.INFO)


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    data_cfg = params["data"]
    train_cfg = params["training"]
    out_cfg = params["output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cifar_batches_dir = str(Path("data") / "cifar-10-batches-py")

    logging.info("Stage: evaluate")
    _, _, test_df = process_data(
        images_dir=str(Path("data") / "images"),
        labels_path=cifar_batches_dir,
        config=data_cfg,
    )

    test_loader = create_data_loader("", test_df, {**train_cfg, "transform": eval_transform})

    model = CifarCNN(n_classes=data_cfg["n_classes"])
    model.load_state_dict(torch.load(out_cfg["save_path"], map_location=device))
    loss_fn = nn.CrossEntropyLoss()

    metrics = test_model(model, test_loader, loss_fn, device)

    metrics_path = Path("outputs/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
