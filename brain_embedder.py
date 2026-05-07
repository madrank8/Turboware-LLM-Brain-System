"""Ollama embedding client with an LRU cache on query embeddings.

Uses httpx.AsyncClient for persistent connections. Identical query strings
hit the cache and skip the round-trip — important for the retrieve path,
which embeds the user's last message every call.
"""
from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Optional, Sequence

import httpx

from brain_config import CONFIG

logger = logging.getLogger(__name__)


class _LRU(OrderedDict):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def get_or_none(self, key: str) -> Optional[list[float]]:
        if key not in self:
            return None
        self.move_to_end(key)
        return self[key]

    def put(self, key: str, value: list[float]) -> None:
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.capacity:
            self.popitem(last=False)


class Embedder:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=CONFIG.ollama_url,
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self._cache = _LRU(CONFIG.embed_cache_size)
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        await self._client.aclose()

    async def embed(self, text: str) -> list[float]:
        text = (text or "").strip()
        if not text:
            return [0.0] * CONFIG.embedding_dim
        cached = self._cache.get_or_none(text)
        if cached is not None:
            return cached
        vec = await self._embed_remote(text)
        self._cache.put(text, vec)
        return vec

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        # Ollama's embedding endpoint is single-input; fan out concurrently.
        return await asyncio.gather(*(self.embed(t) for t in texts))

    async def _embed_remote(self, text: str) -> list[float]:
        payload = {
            "model": CONFIG.embedding_model,
            "prompt": text,
            "keep_alive": CONFIG.embedding_keep_alive,
        }
        try:
            resp = await self._client.post("/api/embeddings", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("embedding call failed: %s", exc)
            return [0.0] * CONFIG.embedding_dim
        data = resp.json()
        vec = data.get("embedding") or []
        if len(vec) != CONFIG.embedding_dim:
            logger.warning(
                "embedding dim mismatch: expected %d, got %d",
                CONFIG.embedding_dim, len(vec),
            )
        return list(vec)


_singleton: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _singleton
    if _singleton is None:
        _singleton = Embedder()
    return _singleton
