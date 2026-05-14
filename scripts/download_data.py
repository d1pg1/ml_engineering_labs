"""DVC stage: download and extract the CIFAR-10 dataset."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.download import download_and_extract

logging.basicConfig(level=logging.INFO)

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CIFAR-10 dataset")
    parser.add_argument("--data-dir", default="data", help="Directory to save data")
    args = parser.parse_args()

    logging.info("Stage: download_data")
    download_and_extract(CIFAR_URL, args.data_dir)
    logging.info("Download stage complete.")


if __name__ == "__main__":
    main()
