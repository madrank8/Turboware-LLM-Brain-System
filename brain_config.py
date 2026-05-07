"""Centralized configuration for the Brain memory system.

All values come from environment variables with sane defaults so the system
boots without a .env in development. Override anything in production via
the LiteLLM service unit's Environment= directives.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _env_list(name: str, default: Sequence[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class BrainConfig:
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://litellm:litellm@localhost:5432/litellm"
    )
    pool_min_size: int = _env_int("BRAIN_PG_POOL_MIN", 5)
    pool_max_size: int = _env_int("BRAIN_PG_POOL_MAX", 15)

    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("BRAIN_EMBED_MODEL", "qwen3-embedding:latest")
    embedding_dim: int = _env_int("BRAIN_EMBED_DIM", 4096)
    embedding_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "24h")
    embed_cache_size: int = _env_int("BRAIN_EMBED_CACHE", 100)

    reranker_model: str = os.getenv(
        "BRAIN_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    rerank_candidates: int = _env_int("BRAIN_RERANK_K", 15)
    rerank_top_k: int = _env_int("BRAIN_TOP_K", 5)
    skill_top_k: int = _env_int("BRAIN_SKILL_TOP_K", 2)
    persona_top_k: int = _env_int("BRAIN_PERSONA_TOP_K", 2)
    persona_keyword_min_overlap: int = _env_int("BRAIN_PERSONA_MIN_OVERLAP", 2)

    retrieval_budget_ms: int = _env_int("BRAIN_RETRIEVAL_BUDGET_MS", 500)
    edge_expansion_depth: int = _env_int("BRAIN_EDGE_DEPTH", 1)

    capture_min_code_lines: int = _env_int("BRAIN_CAPTURE_MIN_LINES", 10)
    capture_summary_markers: list[str] = field(
        default_factory=lambda: _env_list(
            "BRAIN_SUMMARY_MARKERS",
            (
                "design pattern",
                "best practice",
                "the reason is",
                "the rationale",
                "trade-off",
                "tradeoff",
                "convention",
            ),
        )
    )
    learn_problem_markers: list[str] = field(
        default_factory=lambda: _env_list(
            "BRAIN_PROBLEM_MARKERS",
            (
                "error",
                "bug",
                "fix",
                "broken",
                "why does",
                "how do i fix",
                "traceback",
                "exception",
            ),
        )
    )

    skill_repos_url: str = os.getenv(
        "BRAIN_SKILL_REPOS",
        "https://raw.githubusercontent.com/madrank8/turboware-llm-brain-system/main/skills.json",
    )
    persona_repo_owner: str = os.getenv("BRAIN_PERSONA_OWNER", "msitarzewski")
    persona_repo_name: str = os.getenv("BRAIN_PERSONA_REPO", "agency-agents")
    persona_repo_ref: str = os.getenv("BRAIN_PERSONA_REF", "main")

    graphify_binary: str = os.getenv("BRAIN_GRAPHIFY_BIN", "graphify")
    log_level: str = os.getenv("BRAIN_LOG_LEVEL", "INFO")

    cf_heartbeat_seconds: float = _env_float("BRAIN_CF_HEARTBEAT_SECONDS", 20.0)
    cf_streaming_paths: list[str] = field(
        default_factory=lambda: _env_list(
            "BRAIN_CF_STREAM_PATHS", ("/v1/chat/completions", "/v1/messages")
        )
    )


CONFIG = BrainConfig()
