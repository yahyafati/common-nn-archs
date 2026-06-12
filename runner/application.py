import torch
import torch.nn as nn
from torch import optim

from models.registry import get_model
from runner.types import Args
from trainers.trainer import Trainer


class Application:

    def __init__(self, args: Args):
        self.args = args

    def run(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        archs = [a.strip() for a in self.args["arch"].split(",")]

        for arch in archs:
            print(f"\n=== Training {arch} ===")

            model = get_model(arch, num_classes=10)

            optimizer = optim.Adam(model.parameters(), lr=self.args["lr"])
            criterion = nn.CrossEntropyLoss()

            trainer = Trainer(model, optimizer, criterion, device)

            # dummy loader for now
            dummy_loader = [
                (torch.randn(8, 3, 224, 224), torch.randint(0, 10, (8,)))
                for _ in range(10)
            ]

            trainer.fit(dummy_loader, self.args["epochs"])
