from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from PIL import Image


@dataclass
class Message:
    role: str
    text: str = ""
    images: list[Image.Image] = field(default_factory=list)


@dataclass
class GenParams:
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    stop: list[str] = field(default_factory=list)


@dataclass
class Response:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


def system(text: str) -> Message:
    return Message(role="system", text=text)


def user(text: str, images: list[Image.Image] | None = None) -> Message:
    return Message(role="user", text=text, images=list(images or []))


def assistant(text: str) -> Message:
    return Message(role="assistant", text=text)
