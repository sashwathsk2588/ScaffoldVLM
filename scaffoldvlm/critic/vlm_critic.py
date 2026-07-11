from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from PIL import Image
from scaffoldvlm.vlm.messages import system, user, GenParams
from scaffoldvlm.vlm.json_mode import generate_json


_SYSTEM = ("You are a strict layout critic. Return JSON only, no prose, no markdown.")

_USER_TEMPLATE = """Grade this scene layout for the task.

TASK: {task_description}

Return {{"score": float in [0,1], "issues": [str, ...], "directives": [str, ...]}}.
Each directive must be phrased as a constraint addition, e.g.
"dishwasher should be within 60cm of sink" or
"leave 90cm clearance in front of dishwasher for robot approach".
Directives will be appended to the layout criteria for the next solver pass.
"""


@dataclass
class CritiqueResult:
    score: float
    issues: list[str] = field(default_factory=list)
    directives: list[str] = field(default_factory=list)


class VLMCritic:
    def __init__(self, vlm: Any):
        self.vlm = vlm

    def critique(self, renders: dict[str, Image.Image], *,
                 task_description: str) -> CritiqueResult:
        images = list(renders.values())
        msgs = [system(_SYSTEM),
                user(_USER_TEMPLATE.format(task_description=task_description),
                     images=images)]
        data = generate_json(self.vlm, messages=msgs,
                             params=GenParams(temperature=0.2, max_new_tokens=1024))
        return CritiqueResult(
            score=float(data["score"]),
            issues=list(data.get("issues", [])),
            directives=list(data.get("directives", [])),
        )
