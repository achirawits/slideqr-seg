"""Config loading with dotted-key overrides.

Functions in this package never bake config values into argument defaults;
they receive the loaded config (or read it at call time) so ablation
overrides always take effect.
"""
import copy
import json
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "synth.yaml"


def _parse_value(text):
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def set_dotted(cfg, key, value):
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"unknown config key: {key}")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"unknown config key: {key}")
    node[parts[-1]] = value


def load_config(path=None, overrides=None):
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        set_dotted(cfg, key.strip(), _parse_value(raw.strip()))
    return cfg


def dump_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False, allow_unicode=True)


def config_hash(cfg):
    import hashlib
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
