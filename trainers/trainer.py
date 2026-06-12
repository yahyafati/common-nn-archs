import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.logger import get_logger

logger = get_logger(__name__)


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

    def train_one_epoch(self, loader: DataLoader):
        self.model.train()
        total_loss = 0
        num_batches = len(loader)

        logger.debug(f"Starting training epoch with {num_batches} batches")

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss

            if batch_idx % 100 == 0 or batch_idx == num_batches - 1:
                logger.debug(f"Batch [{batch_idx}/{num_batches}] loss={batch_loss:.4f}")

        avg_loss = total_loss / num_batches
        logger.debug(f"Epoch complete: avg_loss={avg_loss:.4f}")

        return avg_loss

    def fit(self, train_loader, epochs):
        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs} started")

            loss = self.train_one_epoch(train_loader)

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
