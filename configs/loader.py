import tomllib

from configs.types import Config, InstanceConfig, DatasetConfig


def load_config(path: str = "config.toml") -> Config:
    raw = tomllib.load(open(path, "rb"))
    return Config(
        title=raw["title"],
        instance=InstanceConfig(run_id_format=raw["instance"]["run_id_format"]),
        dataset=DatasetConfig(root=raw["dataset"]["root"]),
    )
