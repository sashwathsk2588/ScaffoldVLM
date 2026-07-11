"""Port of third_party/layoutvlm/main.py plus a critic refinement loop.

`prepare_task_assets` is a verbatim copy of the vendored helper -- kept in-tree
so we can call it against an in-memory task dict without touching disk.
`Pipeline.run(scene_config)`:
    1. prepare_task_assets(scene_config, asset_dir)
    2. loop up to refinement.max_iters:
        a. layoutvlm_cls(mode=one_shot, ...).solve(scene_config) -> layout
        b. renderer.render(layout, scene_config) -> images
        c. vlm_critic.critique(images, task_description) -> {score, directives}
        d. if score >= threshold: break
        e. append directives to scene_config["layout_criteria"]
    3. write layout.json and return payload
"""
from __future__ import annotations
import collections
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from scaffoldvlm.critic.vlm_critic import VLMCritic


def _slug(s: str) -> str:
    return (s.replace("-", "_").replace(" ", "_").replace("'", "_")
             .replace("/", "_").replace(",", "_").lower())


def prepare_task_assets(task: dict, asset_dir: str) -> dict:
    """Copy of vendored `layoutvlm/main.py:prepare_task_assets`."""
    if "layout_criteria" not in task:
        task["layout_criteria"] = ("the layout should follow the task description "
                                   "and adhere to common sense")

    all_data: dict[str, list] = collections.defaultdict(list)
    for original_uid in list(task["assets"].keys()):
        uid = "-".join(original_uid.split("-")[:-1])
        data_path = os.path.join(asset_dir, uid, "data.json")
        if not os.path.exists(data_path):
            print(f"Warning: Asset data not found for {uid}")
            continue
        with open(data_path, "r") as f:
            data = json.load(f)
        data["path"] = os.path.join(asset_dir, uid, f"{uid}.glb")
        all_data[uid].append(data)

    category_count: dict[str, int] = collections.defaultdict(int)
    for uid, duplicated in all_data.items():
        cat = _slug(duplicated[0]["annotations"]["category"])
        category_count[cat] += 1

    task["assets"] = {}
    category_idx: dict[str, int] = collections.defaultdict(int)

    for uid, duplicated in all_data.items():
        cat = _slug(duplicated[0]["annotations"]["category"])
        category_idx[cat] += 1
        for instance_idx, data in enumerate(duplicated):
            # Diverges from vendored main.py: original reassigns to the outer
            # name inside the inner loop, causing double-suffixing when a
            # category has >1 uid AND >1 instance per uid. We derive cat_var
            # fresh each iteration.
            cat_var = (f"{cat}_{chr(ord('A') + category_idx[cat] - 1)}"
                       if category_count[cat] > 1 else cat)
            var_name = (f"{cat_var}_{instance_idx}"
                        if len(duplicated) > 1 else cat_var)
            task["assets"][f"{cat_var}-{instance_idx}"] = {
                "uid": uid,
                "count": len(duplicated),
                "instance_var_name": var_name,
                "asset_var_name": cat_var,
                "instance_idx": instance_idx,
                "annotations": data["annotations"],
                "category": data["annotations"]["category"],
                "description": data["annotations"]["description"],
                "path": data["path"],
                "onCeiling": data["annotations"]["onCeiling"],
                "onFloor": data["annotations"]["onFloor"],
                "onWall": data["annotations"]["onWall"],
                "onObject": data["annotations"]["onObject"],
                "frontView": data["annotations"]["frontView"],
                "assetMetadata": {
                    "boundingBox": {
                        "x": float(data["assetMetadata"]["boundingBox"]["y"]),
                        "y": float(data["assetMetadata"]["boundingBox"]["x"]),
                        "z": float(data["assetMetadata"]["boundingBox"]["z"]),
                    },
                },
            }
    return task


@dataclass
class Pipeline:
    cfg: Any
    vlm: Any
    layoutvlm_cls: Any = None
    renderer: Any = None
    asset_dir: str = "./objaverse_processed"
    save_dir: str = "./results/scaffoldvlm_run"
    critic: VLMCritic = field(init=False)

    def __post_init__(self):
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.critic = VLMCritic(vlm=self.vlm)
        if self.layoutvlm_cls is None:
            from layoutvlm.layoutvlm import LayoutVLM as _LV
            self.layoutvlm_cls = _LV
        if self.renderer is None:
            self.renderer = _NoopRenderer()

    def run(self, scene_config: dict) -> dict:
        scene_config = prepare_task_assets(dict(scene_config), self.asset_dir)
        base_criteria = scene_config["layout_criteria"]
        accumulated: list[str] = []
        layout: dict = {}
        score = 0.0
        it = 0
        for it in range(self.cfg.refinement.max_iters):
            scene_config["layout_criteria"] = self._compose_criteria(
                base_criteria, accumulated)
            solver = self.layoutvlm_cls(
                mode="one_shot",
                save_dir=str(Path(self.save_dir) / f"iter_{it}"),
                asset_source="objaverse",
            )
            layout = solver.solve(scene_config)
            renders = self.renderer.render(layout, scene_config)
            if not renders:
                score = 0.0
                break
            crit = self.critic.critique(
                renders, task_description=scene_config["task_description"])
            score = crit.score
            if score >= self.cfg.refinement.accept_threshold:
                break
            if not crit.directives:
                break
            accumulated.extend(crit.directives)
        out_path = Path(self.save_dir) / "layout.json"
        payload = {
            "placements": layout,
            "iterations": it + 1,
            "final_score": score,
            "layout_criteria_used": scene_config["layout_criteria"],
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    @staticmethod
    def _compose_criteria(base: str, directives: list[str]) -> str:
        if not directives:
            return base
        return (base + "\n\nAdditional constraints from prior critic passes:\n"
                + "\n".join(f"- {d}" for d in directives))


class _NoopRenderer:
    def render(self, layout, scene_config): return {}
