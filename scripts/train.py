"""DVC stage: train the model using params from params.yaml."""
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_data_loader, eval_transform, process_data, train_transform
from src.models.cnn import CifarCNN
from src.training.trainer import train_model

logging.basicConfig(level=logging.INFO)


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    data_cfg = params["data"]
    train_cfg = params["training"]
    out_cfg = params["output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cifar_batches_dir = str(Path("data") / "cifar-10-batches-py")

    logging.info("Stage: train")
    train_df, val_df, _ = process_data(
        images_dir=str(Path("data") / "images"),
        labels_path=cifar_batches_dir,
        config=data_cfg,
    )

    train_loader = create_data_loader("", train_df, {**train_cfg, "transform": train_transform})
    val_loader = create_data_loader("", val_df, {**train_cfg, "transform": eval_transform})

    model = CifarCNN(n_classes=data_cfg["n_classes"])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])

    save_path = Path(out_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_model(
        model, train_loader, val_loader, loss_fn, optimizer,
        num_epochs=train_cfg["num_epochs"], device=device, save_path=save_path,
    )
    logging.info("Train stage complete.")


if __name__ == "__main__":
    main()
