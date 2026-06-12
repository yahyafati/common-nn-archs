import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from dataloader.cifar10 import CIFAR10Dataset
from models.registry import get_model
from runner.types import Args
from trainers.trainer import Trainer
from utils.logger import get_logger

logger = get_logger()


class Application:

    def __init__(self, args: Args):
        self.args = args

    def run(self):
        device = self.args["device"]
        logger.info(f"Using device: {device}")

        archs = [a.strip() for a in self.args["arch"].split(",")]
        dataset = CIFAR10Dataset(
            root="data/cifar-10-batches-py",
            mode=["train", "validation", "test"],
            download=True,
        )

        for arch in archs:
            logger.info(f"=== Training {arch} ===")

            model = get_model(arch, num_classes=10)

            optimizer = optim.Adam(model.parameters(), lr=self.args["lr"])
            criterion = nn.CrossEntropyLoss()

            trainer = Trainer(model, optimizer, criterion, device)

            dataloader = DataLoader(
                dataset,
                batch_size=self.args["batch_size"],
                shuffle=True,
            )

            trainer.fit(dataloader, self.args["epochs"])
