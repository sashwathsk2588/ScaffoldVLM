from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints
import yaml

DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"


@dataclass
class LoopCfg:
    task_region_max_iters: int
    decor_region_max_iters: int
    accept_threshold: float
    catastrophic_threshold: float


@dataclass
class CriticWeights:
    visual: float
    reachability: float
    physics: float


@dataclass
class CriticCfg:
    weights: CriticWeights
    render_viewpoints: list[str]
    physics_settle_frames: int
    physics_drift_cutoff_cm: float


@dataclass
class VLMCfg:
    backend: str
    model: str
    endpoint: str | None
    max_retries: int


@dataclass
class AssetsCfg:
    objaverse_root: str
    partnet_root: str
    usd_cache: str


@dataclass
class IsaacCfg:
    enabled: str


@dataclass
class RefinementCfg:
    max_iters: int
    accept_threshold: float


@dataclass
class Config:
    loop: LoopCfg
    critic: CriticCfg
    vlm: VLMCfg
    assets: AssetsCfg
    isaac: IsaacCfg
    refinement: RefinementCfg


def _from_dict(cls, data: dict) -> Any:
    if not is_dataclass(cls):
        return data
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            raise ValueError(f"Missing config key: {f.name}")
        ftype = hints[f.name]
        val = data[f.name]
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(ftype, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            raise ValueError(f"Unknown config key: {key}")
        cur = cur[p]
    if parts[-1] not in cur:
        raise ValueError(f"Unknown config key: {key}")
    cur[parts[-1]] = value


def load_config(config_path: Path | None = None,
                overrides: dict[str, Any] | None = None) -> Config:
    with open(DEFAULTS_PATH) as f:
        data = yaml.safe_load(f)
    if config_path is not None:
        with open(config_path) as f:
            user = yaml.safe_load(f) or {}
        data = _deep_merge(data, user)
    for k, v in (overrides or {}).items():
        _apply_dotted(data, k, v)
    return _from_dict(Config, data)
