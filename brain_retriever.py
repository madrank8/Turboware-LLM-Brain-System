"""Multi-stage retrieval pipeline.

Pipeline (target ~260ms warm, 500ms budget):
  embed query -> persona keyword/vector match (parallel)
              -> per-table vector + FTS (parallel)
              -> merge, dedup by (table, entry_id), recency boost
              -> CrossEncoder rerank top-K
              -> trim to top_k, edge expand for neighbors
              -> independent skill search appended in dedicated slots

Any error in any stage is logged and produces an empty result for that stage —
the caller proceeds without context rather than blocking the user request.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from brain_config import CONFIG
import brain_db as db
from brain_embedder import get_embedder

logger = logging.getLogger(__name__)


_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "we", "it", "this", "that", "these", "those", "do", "does",
    "did", "have", "has", "had", "will", "would", "could", "should", "can",
    "may", "might", "must", "shall", "what", "how", "why", "when", "where",
    "who", "which", "if", "then", "else", "as", "than", "so", "not", "no",
    "yes", "please", "help", "code", "use", "using",
}


def extract_keywords(text: str, limit: int = 12) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok in _STOP_WORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# CrossEncoder reranker (lazy, background-loaded)
# ---------------------------------------------------------------------------

class _Reranker:
    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.Lock()
        self._load_started = False

    def warm(self) -> None:
        with self._lock:
            if self._load_started:
                return
            self._load_started = True
        threading.Thread(target=self._load, daemon=True, name="brain-rerank-warm").start()

    def _load(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(CONFIG.reranker_model)
            logger.info("reranker loaded: %s", CONFIG.reranker_model)
        except Exception as exc:
            logger.warning("reranker load failed: %s", exc)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        if not self.ready:
            return [0.0] * len(candidates)
        pairs = [(query, c) for c in candidates]
        try:
            return [float(s) for s in self._model.predict(pairs)]
        except Exception as exc:
            logger.warning("rerank failed: %s", exc)
            return [0.0] * len(candidates)


_RERANKER = _Reranker()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    personas: list[dict[str, Any]] = field(default_factory=list)
    contexts: list[dict[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retrieval entry point
# ---------------------------------------------------------------------------

async def retrieve(query: str, *, code_blocks: Optional[list[str]] = None) -> RetrievalResult:
    _RERANKER.warm()
    start = time.perf_counter()
    timings: dict[str, float] = {}

    full_query = query
    if code_blocks:
        full_query = query + "\n\n" + "\n\n".join(code_blocks)

    embedder = get_embedder()
    t0 = time.perf_counter()
    embedding = await embedder.embed(full_query[:8000])
    timings["embed_ms"] = (time.perf_counter() - t0) * 1000

    keywords = extract_keywords(query)

    try:
        async with asyncio.timeout(CONFIG.retrieval_budget_ms / 1000):
            persona_task = asyncio.create_task(_retrieve_personas(embedding, keywords))
            context_task = asyncio.create_task(_retrieve_contexts(query, embedding))
            skills_task = asyncio.create_task(
                db.skill_vector_search(embedding, CONFIG.skill_top_k)
            )
            personas, contexts, skills = await asyncio.gather(
                persona_task, context_task, skills_task
            )
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning("retrieval exceeded %dms budget", CONFIG.retrieval_budget_ms)
        return RetrievalResult(timings_ms={"total_ms": (time.perf_counter() - start) * 1000})
    except Exception as exc:
        logger.warning("retrieval failed: %s", exc)
        return RetrievalResult(timings_ms={"total_ms": (time.perf_counter() - start) * 1000})

    contexts = await _rerank_and_trim(query, contexts)
    contexts = await _expand_edges(contexts)

    asyncio.create_task(_touch_async(contexts))

    timings["total_ms"] = (time.perf_counter() - start) * 1000
    return RetrievalResult(
        personas=personas, contexts=contexts, skills=skills, timings_ms=timings
    )


async def _retrieve_personas(embedding: Sequence[float], keywords: Sequence[str]) -> list[dict[str, Any]]:
    if keywords:
        kw_hits = await db.persona_keyword_search(keywords, CONFIG.persona_top_k)
        if kw_hits and kw_hits[0]["score"] >= CONFIG.persona_keyword_min_overlap:
            return kw_hits
    return await db.persona_vector_search(embedding, CONFIG.persona_top_k)


async def _retrieve_contexts(query: str, embedding: Sequence[float]) -> list[dict[str, Any]]:
    tasks: list[asyncio.Task] = []
    for table in db.VECTOR_TABLES:
        tasks.append(asyncio.create_task(db.vector_search(table, embedding, 10)))
        tasks.append(asyncio.create_task(db.fts_search(table, query, 5)))
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for result in raw:
        if isinstance(result, Exception):
            logger.warning("search shard failed: %s", result)
            continue
        for hit in result:
            key = (hit["table"], hit["entry_id"])
            existing = merged.get(key)
            # Vector results (higher base weight) win over FTS for identical entries.
            if existing is None or hit["score"] > existing["score"]:
                merged[key] = hit
    return list(merged.values())


async def _rerank_and_trim(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not hits:
        return []
    hits.sort(key=lambda h: h["score"], reverse=True)
    candidates = hits[: CONFIG.rerank_candidates]
    if _RERANKER.ready and candidates:
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None, _RERANKER.score, query, [c["body"] or c["title"] or "" for c in candidates]
        )
        for c, s in zip(candidates, scores):
            c["rerank_score"] = s
        candidates.sort(key=lambda h: h.get("rerank_score", 0.0), reverse=True)
    return candidates[: CONFIG.rerank_top_k]


async def _expand_edges(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not hits or CONFIG.edge_expansion_depth <= 0:
        return hits
    seen = {(h["table"], h["entry_id"]) for h in hits}
    extras: list[dict[str, Any]] = []
    for hit in list(hits):
        try:
            neighbors = await db.get_neighbors(
                hit["table"], hit["entry_id"], CONFIG.edge_expansion_depth
            )
        except Exception as exc:
            logger.debug("edge expand failed: %s", exc)
            continue
        for n in neighbors:
            key = (n["table"], n["entry_id"])
            if key in seen:
                continue
            seen.add(key)
            full = await db.hydrate(n["table"], n["entry_id"])
            if full is None:
                continue
            extras.append({
                "table": n["table"],
                "entry_id": n["entry_id"],
                "title": full.get("symbol_name") or full.get("title") or full.get("problem", ""),
                "body": full.get("content") or full.get("solution", ""),
                "score": hit["score"] * 0.5,
                "source": f"edge:{n['relation_type']}",
            })
    return hits + extras


async def _touch_async(hits: list[dict[str, Any]]) -> None:
    by_table: dict[str, list[int]] = {}
    for h in hits:
        by_table.setdefault(h["table"], []).append(h["entry_id"])
    for table, ids in by_table.items():
        try:
            await db.touch_access(table, ids)
        except Exception as exc:
            logger.debug("touch_access failed: %s", exc)
