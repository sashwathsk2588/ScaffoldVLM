"""Route vendored Holodeck / LayoutVLM langchain LLM calls through our VLM.

Vendored source is NOT edited. Class-level swap: instances built AFTER
`install_vlm_shims` are shims; earlier instances aren't. Call install
BEFORE constructing LayoutVLM.
"""
from __future__ import annotations
import importlib
from typing import Any
from .base import VLM
from .messages import Message, GenParams, system, user, assistant


_HOLODECK_TARGETS: list[tuple[str, str]] = [
    ("ai2holodeck.generation.rooms", "OpenAI"),
    ("ai2holodeck.generation.floor_objects", "OpenAI"),
    ("ai2holodeck.generation.object_selector", "OpenAI"),
    ("ai2holodeck.generation.windows", "OpenAI"),
    ("ai2holodeck.generation.small_objects", "OpenAI"),
    ("ai2holodeck.generation.walls", "OpenAI"),
    ("ai2holodeck.generation.holodeck", "OpenAI"),
    ("ai2holodeck.generation.doors", "OpenAI"),
    ("ai2holodeck.generation.wall_objects", "OpenAI"),
    ("ai2holodeck.generation.ceiling_objects", "OpenAI"),
]
_LAYOUTVLM_TARGETS: list[tuple[str, str]] = [
    ("layoutvlm.layoutvlm", "ChatOpenAI"),
]


class _Response:
    def __init__(self, content: str):
        self.content = content


class _LangchainLLMShim:
    _vlm: VLM = None

    def __init__(self, *args: Any, **kwargs: Any):
        self._params = GenParams(
            max_new_tokens=int(kwargs.get("max_tokens", 1024)),
            temperature=float(kwargs.get("temperature", 0.2)),
            top_p=float(kwargs.get("top_p", 0.95)),
        )

    @classmethod
    def bind(cls, vlm: VLM) -> type:
        return type("_BoundLangchainShim", (cls,), {"_vlm": vlm})

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        return self._vlm.generate([user(prompt)], self._params).text

    def invoke(self, messages: Any, *args, **kwargs) -> _Response:
        return _Response(
            self._vlm.generate(_convert_langchain_messages(messages), self._params).text
        )


def _convert_langchain_messages(messages: Any) -> list[Message]:
    out: list[Message] = []
    for m in messages:
        role = _role_of(m)
        text = _text_of(m)
        if role == "system":
            out.append(system(text))
        elif role == "assistant":
            out.append(assistant(text))
        else:
            out.append(user(text))
    return out


def _role_of(m: Any) -> str:
    t = getattr(m, "type", None)
    if t == "system":
        return "system"
    if t in ("ai", "assistant"):
        return "assistant"
    return "user"


def _text_of(m: Any) -> str:
    c = getattr(m, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(c)


def install_vlm_shims(vlm: VLM) -> None:
    shim_cls = _LangchainLLMShim.bind(vlm)
    for mod_path, attr in _HOLODECK_TARGETS + _LAYOUTVLM_TARGETS:
        try:
            mod = importlib.import_module(mod_path)
        except Exception:
            continue
        if hasattr(mod, attr):
            setattr(mod, attr, shim_cls)
