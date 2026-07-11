from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from .base import VLM
from .messages import Message, GenParams, Response


@dataclass
class _Call:
    messages: list[Message]
    params: GenParams


class FixtureVLM(VLM):
    """Queue-based fixture replay for tests. Prime with .queue(label, text)
    or .queue_file(name) which reads <name>.json's 'response' field."""

    def __init__(self, fixtures_dir: Path | None = None):
        self.dir = Path(fixtures_dir) if fixtures_dir else None
        self._queue: list[tuple[str, str]] = []
        self.calls: list[_Call] = []
        self.responses: list[Response] = []

    def queue(self, label: str, text: str) -> None:
        self._queue.append((label, text))

    def queue_file(self, name: str) -> None:
        if self.dir is None:
            raise RuntimeError("FixtureVLM has no fixtures_dir; use queue()")
        path = self.dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing fixture: {path}")
        with open(path) as f:
            self._queue.append((name, json.load(f)["response"]))

    def generate(self, messages, params):
        if not self._queue:
            raise RuntimeError("FixtureVLM queue is empty; call queue() first")
        _, text = self._queue.pop(0)
        r = Response(text=text)
        self.calls.append(_Call(messages=list(messages), params=params))
        self.responses.append(r)
        return r
