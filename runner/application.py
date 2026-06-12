from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from dataloader.cifar10 import CIFAR10Dataset
from models.registry import get_model
from runner.types import Context
from trainers.trainer import Trainer
from utils.instance import generate_run_id
from utils.logger import get_logger

logger = get_logger()


class Application:

    def __init__(self, context: Context):
        self.context = context
        self.run_id = generate_run_id(self.context.config.instance.run_id_format)
        logger.info(f"Run ID: {self.run_id}")
        self.models: dict[str, nn.Module] = {}

    def run(self):
        args = self.context.args
        device = args.device
        logger.info(f"Using device: {device}")

        archs = [a.strip() for a in args.arch.split(",")]
        train_dataset = CIFAR10Dataset(
            root="data/cifar-10-batches-py",
            mode=["train"],
            download=True,
        )
        test_dataset = CIFAR10Dataset(
            root="data/cifar-10-batches-py",
            mode=["test"],
            download=True,
        )
        val_dataset = CIFAR10Dataset(
            root="data/cifar-10-batches-py",
            mode=["validation"],
            download=True,
        )

        for arch in archs:
            logger.info(f"=== Training {arch} ===")

            model = get_model(arch, num_classes=10)
            self.models[arch] = model
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.lr / 100.0,
            )

            criterion = nn.CrossEntropyLoss()

            trainer = Trainer(model, optimizer, criterion, device)

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=True
            )

            trainer.fit(train_dataloader, args.epochs)
            trainer.validate(val_dataloader)

    def save(self):
        """Save all registered models to the run-specific folder."""

        save_dir = Path(self.context.args.save_dir)
        run_dir = save_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(self.models)} model(s) to {run_dir}")

        saved_paths = []
        for name, model in self.models.items():
            safe_name = name.replace("/", "_").replace("\\", "_")
            filepath = run_dir / f"{safe_name}.pt"
            torch.save(model.state_dict(), filepath)
            saved_paths.append(str(filepath))
            logger.debug(f"Saved model '{name}' -> {filepath}")

        logger.info(f"Successfully saved {len(saved_paths)} model(s)")
        return saved_paths
