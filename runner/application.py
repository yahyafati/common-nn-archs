import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from dataloader.cifar10 import CIFAR10Dataset
from models.registry import get_model
from runner.types import Context
from trainers.trainer import Trainer, DataPoint
from utils.instance import generate_run_id
from utils.logger import get_logger

logger = get_logger()


class Application:

    def __init__(self, context: Context):
        self.context = context
        self.run_id = generate_run_id(self.context.config.instance.run_id_format)
        logger.info(f"Run ID: {self.run_id}")
        self.models: dict[str, nn.Module] = {}
        self.accuracies: dict[str, list[DataPoint]] = {}

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

            acc = trainer.fit(train_dataloader, val_dataloader, args.epochs)
            self.accuracies[arch] = acc

    def save_accuracies_to_csv(
        self,
        filepath: str | Path,
        mode: str = "w",
    ) -> None:
        """
        Save accuracy data to a CSV file.

        Args:
            filepath: Path to the output CSV file.
            mode: File open mode. 'w' to overwrite, 'a' to append.
        """
        logger.info(f"Saving accuracy to {filepath}")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "key",
            "epoch",
            "train_accuracy",
            "val_top1_accuracy",
            "val_top3_accuracy",
        ]
        file_exists = filepath.exists() and filepath.stat().st_size > 0

        with open(filepath, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if mode == "w" or not file_exists:
                writer.writeheader()

            for key, data_points in self.accuracies.items():
                for epoch, dp in enumerate(data_points, start=1):
                    row = {"key": key, "epoch": epoch, **asdict(dp)}
                    writer.writerow(row)

        logger.info(f"Saved accuracy to {filepath}")

    def save_accuracy_plot(
        self,
        filepath: str | Path,
        figsize: tuple[int, int] = (12, 6),
        dpi: int = 300,
        title: str = "Training & Validation Accuracies",
        show_top3: bool = True,
    ) -> None:
        """
        Save a plot of accuracies to an image file.

        Args:
            filepath: Path to the output image file (e.g., 'plot.png', 'plot.pdf').
            figsize: Figure size in inches (width, height).
            dpi: Resolution for raster formats (png, jpg).
            title: Plot title.
            show_top3: Whether to include val_top3_accuracy lines.
        """
        logger.info(f"Saving accuracy plot to {filepath}")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        colors = plt.cm.tab10.colors

        for idx, (key, data_points) in enumerate(self.accuracies.items()):
            color = colors[idx % len(colors)]
            epochs = list(range(1, len(data_points) + 1))
            train_acc = [dp.train_accuracy for dp in data_points]
            val_top1 = [dp.val_top1_accuracy for dp in data_points]
            val_top3 = [dp.val_top3_accuracy for dp in data_points]

            # Left subplot: Training accuracy
            axes[0].plot(
                epochs, train_acc, label=key, color=color, marker="o", markersize=4
            )

            # Right subplot: Validation accuracy
            axes[1].plot(
                epochs,
                val_top1,
                label=f"{key} (top-1)",
                color=color,
                linestyle="-",
                marker="s",
                markersize=4,
            )
            if show_top3:
                axes[1].plot(
                    epochs,
                    val_top3,
                    label=f"{key} (top-3)",
                    color=color,
                    linestyle="--",
                    marker="^",
                    markersize=4,
                )

        # Configure left subplot
        axes[0].set_title("Training Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)

        # Configure right subplot
        axes[1].set_title("Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved accuracy plot to {filepath}")

    def save(self):
        """Save all registered models to the run-specific folder."""

        save_dir = Path(self.context.args.save_dir)
        run_dir = save_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self.save_accuracies_to_csv(run_dir / "accuracies.csv")
        self.save_accuracy_plot(run_dir / "plot.png")

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
