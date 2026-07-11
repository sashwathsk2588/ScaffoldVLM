from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .config.loader import load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scaffoldvlm")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate",
                       help="Solve a LayoutVLM scene with critic refinement")
    g.add_argument("--scene-json", required=True,
                   help="Path to the scene JSON (LayoutVLM's format)")
    g.add_argument("--asset-dir", required=True,
                   help="Directory with per-uid <uid>/data.json and <uid>.glb")
    g.add_argument("--save-dir", required=True)
    g.add_argument("--config", default=None)
    g.add_argument("--set", action="append", default=[])
    return p


def _parse_overrides(items: list[str]) -> dict:
    def _cast(v: str):
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v
    return {k: _cast(v) for k, v in (i.split("=", 1) for i in items)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd != "generate":
        return 2
    cfg = load_config(config_path=Path(args.config) if args.config else None,
                      overrides=_parse_overrides(args.set))
    _run(cfg, scene_json=args.scene_json, asset_dir=args.asset_dir,
         save_dir=args.save_dir)
    return 0


def _run(cfg, *, scene_json: str, asset_dir: str, save_dir: str) -> None:
    from .vlm.qwen3vl import build_vlm
    from .vlm.adapters import install_vlm_shims
    from .pipeline import Pipeline
    vlm = build_vlm(cfg)
    install_vlm_shims(vlm)
    with open(scene_json) as f:
        scene_config = json.load(f)
    Pipeline(cfg=cfg, vlm=vlm, asset_dir=asset_dir, save_dir=save_dir).run(scene_config)


if __name__ == "__main__":
    sys.exit(main())
