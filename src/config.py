"""Configuration loading with simple dotted command-line overrides."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import yaml


def _parse_value(text: str):
    """Turn a command-line string into int / float / bool / None / list when possible."""
    return yaml.safe_load(text)


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


def backbone_config(cfg: dict, backbone: str) -> dict:
    """Merge the per-backbone overrides into a copy of the train section."""
    train = copy.deepcopy(cfg["train"])
    train.update(cfg.get("backbones", {}).get(backbone, {}))
    return train


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def run_dir(cfg: dict, name: str) -> Path:
    d = Path(cfg["output_dir"]) / name
    d.mkdir(parents=True, exist_ok=True)
    return d
