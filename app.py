import argparse

from runner.application import Application
from runner.types import Args
from utils.logger import get_logger

logger = get_logger()


def parse_args() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arch", type=str, required=True, help="comma-separated model names"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    parsed = parser.parse_args()
    return Args(
        arch=parsed.arch,
        epochs=parsed.epochs,
        lr=parsed.lr,
    )


def main():
    args = parse_args()

    app = Application(args)
    app.run()


if __name__ == "__main__":
    main()
