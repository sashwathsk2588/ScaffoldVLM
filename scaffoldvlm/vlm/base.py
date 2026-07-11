from __future__ import annotations
from abc import ABC, abstractmethod
from .messages import Message, GenParams, Response


class VLM(ABC):
    """Messages-based VLM. Backends implement generate(messages, params) -> Response."""

    @abstractmethod
    def generate(self, messages: list[Message], params: GenParams) -> Response:
        ...
