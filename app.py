import argparse

import torch

from runner.application import Application
from runner.types import Args
from utils.logger import get_logger

logger = get_logger()


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arch", type=str, required=True, help="comma-separated model names"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        choices=["cuda", "mps", "cpu"],
        help="device to use for training",
    )
    parser.add_argument("--batch-size", type=int, default=64)

    parsed = parser.parse_args()
    return Args(
        arch=parsed.arch,
        epochs=parsed.epochs,
        lr=parsed.lr,
        device=parsed.device,
        batch_size=parsed.batch_size,
    )


def main():
    args = parse_args()

    app = Application(args)
    app.run()


if __name__ == "__main__":
    main()
