from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievedAsset:
    uid: str
    score: float
    source: str = "objaverse"


class ObjaverseRetriever:
    """Thin wrapper over Holodeck's Objaverse retriever.

    `upstream` is either an ai2holodeck.generation.objaverse_retriever
    ObjathorRetriever instance or any object exposing .retrieve(query, top_k).
    """

    def __init__(self, upstream: Any):
        self.upstream = upstream

    @classmethod
    def from_config(cls, cfg):
        from ai2holodeck.generation.objaverse_retriever import ObjathorRetriever
        return cls(upstream=ObjathorRetriever())

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedAsset]:
        raw = self.upstream.retrieve(query, top_k=top_k)
        return [RetrievedAsset(uid=uid, score=float(score)) for uid, score in raw]
