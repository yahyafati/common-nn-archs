import os
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.logger import get_logger

DatasetMode = Literal["train", "test", "validation"]

logger = get_logger()


class CIFAR10Dataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        mode: list[DatasetMode] | DatasetMode = "train",
        transform=None,
        target_transform=None,
        download=False,
    ):
        """
        Args:
            root (str): path to cifar-10-batches-py
            mode (DatasetMode): whether to load training set or test set
            transform (callable, optional): applied to image
            target_transform (callable, optional): applied to label
            download (bool): download if the root folder doesn't exist.
        """
        self.root = root
        self.mode = mode if isinstance(mode, list) else [mode]
        self.transform = transform
        self.target_transform = target_transform

        self.data = []
        self.targets = []

        batch_files = []
        if "train" in self.mode:
            batch_files += [f"data_batch_{i}" for i in range(1, 5)]
        if "validation" in self.mode:
            batch_files += ["data_batch_5"]
        if "test" in self.mode:
            batch_files += ["test_batch"]

        logger.info(f"Initializing CIFAR10 dataset for modes: {self.mode}")

        if download and (not os.path.exists(self.root) or not os.listdir(self.root)):
            _download_cifar10_dataset(str(Path(root).parent))

        for file_name in batch_files:
            path = os.path.join(self.root, file_name)
            logger.debug(f"Loading batch file: {path}")
            try:
                entry = self._unpickle(path)
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
            except FileNotFoundError as e:
                logger.error(
                    f"Failed to find batch file {path}. Did you set download=True?"
                )
                raise e

        self.data = np.vstack(self.data)  # shape: (N, 3072)
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = torch.tensor(self.data, dtype=torch.float32) / 255.0
        self.targets = torch.tensor(self.targets, dtype=torch.long)

        meta_path = os.path.join(self.root, "batches.meta")
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f, encoding="latin1")
                self.classes = meta["label_names"]
            logger.info(
                f"Successfully loaded CIFAR10 dataset with {len(self.data)} samples and classes: {self.classes}"
            )
        except FileNotFoundError:
            logger.warning(
                f"Metadata file not found at {meta_path}. Class names will not be available."
            )
            self.classes = []

    @staticmethod
    def _unpickle(path) -> dict:
        with open(path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        return entry

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target


def _download_cifar10_dataset(root: str):
    import os
    import requests
    import tarfile
    from tqdm import tqdm

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    os.makedirs(root, exist_ok=True)

    archive_path = os.path.join(root, "cifar-10-python.tar.gz")
    extract_path = os.path.join(root, "cifar-10-batches-py")

    if os.path.isdir(extract_path):
        logger.info("CIFAR-10 already downloaded and extracted.")
        return extract_path

    if not os.path.isfile(archive_path):
        logger.info(f"Downloading CIFAR-10 from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to download CIFAR-10 dataset: {e}")
            raise e

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 8192

        with open(archive_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    else:
        logger.info("Archive already exists, skipping download.")

    # Extract
    logger.info("Extracting CIFAR-10...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=root)
        logger.info("Extraction complete.")
    except (tarfile.TarError, EOFError) as e:
        logger.error(f"Corrupted archive file. Failed to extract CIFAR-10: {e}")
        if os.path.exists(archive_path):
            os.remove(archive_path)
        raise e

    # Optional: remove archive to save space
    try:
        os.remove(archive_path)
        logger.debug("Temporary download archive removed.")
    except OSError as e:
        logger.warning(f"Could not remove temporary archive {archive_path}: {e}")

    return extract_path
