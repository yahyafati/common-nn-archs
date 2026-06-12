import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from models.registry import get_model
from trainers.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arch", type=str, required=True, help="comma-separated model names"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    archs = [a.strip() for a in args.arch.split(",")]

    for arch in archs:
        print(f"\n=== Training {arch} ===")

        model = get_model(arch, num_classes=10)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(model, optimizer, criterion, device)

        # dummy loader for now
        dummy_loader = [
            (torch.randn(8, 3, 224, 224), torch.randint(0, 10, (8,))) for _ in range(10)
        ]

        trainer.fit(dummy_loader, args.epochs)


if __name__ == "__main__":
    main()
