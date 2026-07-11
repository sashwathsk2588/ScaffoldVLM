from __future__ import annotations
import json
import re
from typing import Any
from .messages import Message, GenParams, assistant, user


class ExtractError(ValueError):
    pass


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_BARE_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)


def extract_json(raw: str) -> Any:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = _BARE_OBJ_RE.search(text)
    if m:
        candidate = m.group(1)
        depth = 0
        end = None
        for i, ch in enumerate(candidate):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is not None:
            try:
                return json.loads(candidate[:end])
            except json.JSONDecodeError:
                pass
    raise ExtractError(f"could not extract JSON from: {raw[:200]}...")


def generate_json(vlm,
                  *,
                  messages: list[Message],
                  params: GenParams | None,
                  max_repairs: int = 2) -> Any:
    params = params or GenParams()
    convo = list(messages)
    for attempt in range(max_repairs + 1):
        resp = vlm.generate(convo, params)
        try:
            return extract_json(resp.text)
        except ExtractError as e:
            if attempt == max_repairs:
                raise
            convo = convo + [
                assistant(resp.text),
                user(f"Your previous response could not be parsed as JSON ({e}). "
                     "Respond with valid JSON only, no prose, no markdown fences."),
            ]
    raise ExtractError("unreachable")
