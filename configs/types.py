from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    title: str
    instance: "InstanceConfig"
    dataset: "DatasetConfig"


@dataclass(frozen=True)
class InstanceConfig:
    run_id_format: str


@dataclass(frozen=True)
class DatasetConfig:
    root: str
