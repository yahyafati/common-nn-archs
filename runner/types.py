from typing import TypedDict


class Args(TypedDict):
    arch: str
    epochs: int
    lr: float
    device: str
    batch_size: int
