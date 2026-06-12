from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.logger import get_logger
from utils.metrics import MetricTracker, accuracy, MetricTrackerResult

logger = get_logger(__name__)


@dataclass
class DataPoint:
    train_accuracy: float
    val_top1_accuracy: float
    val_top3_accuracy: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Log initialization details
        logger.info(
            f"Trainer initialized: device={device}, "
            f"model={model.__class__.__name__}, "
            f"optimizer={optimizer.__class__.__name__}, "
            f"criterion={criterion.__class__.__name__}"
        )
        logger.info(
            "Optimizer Defaults: "
            + ", ".join([f"{k}: {v}" for k, v in optimizer.defaults.items()])
        )

    def train_one_epoch(self, loader: DataLoader) -> MetricTrackerResult:
        self.model.train()
        num_batches = len(loader)

        tracker = MetricTracker("loss", "accuracy")
        logger.debug(f"Starting training epoch with {num_batches} batches")

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            (batch_accuracy,) = accuracy(out, y)
            batch_loss = loss.item()
            tracker.update(loss=batch_loss, accuracy=batch_accuracy)
        results = tracker.compute()
        avg_loss = results["loss"]
        logger.debug(f"Epoch complete: avg_loss={avg_loss:.4f}")

        return results

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs):
        logger.info(f"Starting training for {epochs} epochs")
        logger.debug("Setting model to training mode")
        self.model.train()

        accuracies: list[DataPoint] = []
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs} started")

            epoch_train_results = self.train_one_epoch(train_loader)
            loss = epoch_train_results["loss"]
            epoch_val_results = self.validate(val_loader)
            accuracies.append(
                DataPoint(
                    train_accuracy=epoch_train_results["accuracy"],
                    val_top1_accuracy=epoch_val_results["top1_accuracy"],
                    val_top3_accuracy=epoch_val_results["top3_accuracy"],
                )
            )
            log_msg = f"Epoch {epoch + 1}/{epochs} complete: avg_loss={loss:.6f}"

            if loss != loss:
                logger.error(
                    f"NaN loss detected at epoch {epoch + 1}! Stopping training."
                )
                raise RuntimeError(f"Training diverged at epoch {epoch + 1} (NaN loss)")
            elif loss < 0:
                logger.warning(f"Negative loss detected: {loss:.6f}")
                logger.info(log_msg)
            else:
                logger.info(log_msg)

        logger.info("Training completed successfully")
        return accuracies

    def validate(self, dataloader: DataLoader) -> MetricTrackerResult:
        logger.debug("Setting model to evaluation mode")
        self.model.eval()

        tracker = MetricTracker("loss", "top1_accuracy", "top3_accuracy")

        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                pred = self.model(features)
                loss = self.criterion(pred, labels)

                acc1, acc3 = accuracy(pred, labels, topk=(1, 3))

                tracker.update(loss=loss.item(), top1_accuracy=acc1, top3_accuracy=acc3)

        results = tracker.compute()

        logger.info(
            f"Validation/Test Error: "
            f"Top-1 Accuracy: {(100 * results['top1_accuracy']):>0.1f}%, "
            f"Top-3 Accuracy: {(100 * results['top3_accuracy']):>0.1f}%, "
            f"Avg loss: {results['loss']:>8f}"
        )

        return results
