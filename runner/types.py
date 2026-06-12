from dataclasses import dataclass

from configs.types import Config


@dataclass(frozen=True)
class Context:
    args: "Args"
    config: Config


@dataclass(frozen=True)
class Args:
    config_path: str
    arch: str
    epochs: int
    lr: float
    device: str
    batch_size: int
    save_dir: str
