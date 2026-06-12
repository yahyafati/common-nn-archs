import argparse

import torch

from configs.loader import load_config
from runner.application import Application
from runner.types import Args, Context
from utils.logger import get_logger

logger = get_logger(__name__)


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config.toml")
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
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    parsed = parser.parse_args()
    return Args(
        config_path=parsed.config,
        arch=parsed.arch,
        epochs=parsed.epochs,
        lr=parsed.lr,
        device=parsed.device,
        batch_size=parsed.batch_size,
        save_dir=parsed.save_dir,
    )


def main():
    args = parse_args()
    config = load_config(args.config_path)
    context = Context(args, config)

    app = Application(context)
    app.run()
    app.save()


if __name__ == "__main__":
    main()
