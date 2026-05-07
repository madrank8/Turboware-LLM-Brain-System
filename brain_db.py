"""PostgreSQL + pgvector access layer for the Brain memory system.

Owns schema initialization for the six brain tables, the asyncpg pool, and the
hybrid search primitives used by the retrieval pipeline. All public functions
are async and assume the pool has been initialized via `init_pool()`.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from brain_config import CONFIG

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def init_pool() -> asyncpg.Pool:
    global _pool
    if _pool is not None:
        return _pool
    _pool = await asyncpg.create_pool(
        dsn=CONFIG.database_url,
        min_size=CONFIG.pool_min_size,
        max_size=CONFIG.pool_max_size,
        init=_register_pgvector_codec,
    )
    await init_schema()
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def acquire():
    if _pool is None:
        await init_pool()
    assert _pool is not None
    async with _pool.acquire() as conn:
        yield conn


async def _register_pgvector_codec(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec(
        "vector",
        encoder=lambda v: "[" + ",".join(f"{x:.8f}" for x in v) + "]",
        decoder=lambda s: [float(x) for x in s.strip("[]").split(",")] if s else [],
        schema="public",
        format="text",
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS brain_code (
    entry_id BIGSERIAL PRIMARY KEY,
    repo_path TEXT NOT NULL,
    file_path TEXT NOT NULL,
    language TEXT,
    symbol_name TEXT,
    symbol_type TEXT,
    content TEXT NOT NULL,
    summary TEXT,
    embedding vector({CONFIG.embedding_dim}),
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (repo_path, file_path, symbol_name)
);
CREATE INDEX IF NOT EXISTS brain_code_fts
    ON brain_code USING gin (to_tsvector('english', coalesce(symbol_name,'') || ' ' || coalesce(summary,'') || ' ' || content));
CREATE INDEX IF NOT EXISTS brain_code_recency ON brain_code (last_accessed_at DESC NULLS LAST);

CREATE TABLE IF NOT EXISTS brain_summary (
    entry_id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    keywords TEXT[],
    embedding vector({CONFIG.embedding_dim}),
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS brain_summary_fts
    ON brain_summary USING gin (to_tsvector('english', title || ' ' || content));
CREATE INDEX IF NOT EXISTS brain_summary_recency ON brain_summary (last_accessed_at DESC NULLS LAST);

CREATE TABLE IF NOT EXISTS brain_experience (
    entry_id BIGSERIAL PRIMARY KEY,
    context_type TEXT NOT NULL,
    problem TEXT NOT NULL,
    solution TEXT NOT NULL,
    outcome TEXT,
    embedding vector({CONFIG.embedding_dim}),
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS brain_experience_fts
    ON brain_experience USING gin (to_tsvector('english', problem || ' ' || solution || ' ' || coalesce(outcome,'')));

CREATE TABLE IF NOT EXISTS brain_skills (
    entry_id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    body TEXT NOT NULL,
    source_repo TEXT,
    source_url TEXT,
    embedding vector({CONFIG.embedding_dim}),
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS brain_skills_fts
    ON brain_skills USING gin (to_tsvector('english', name || ' ' || coalesce(description,'')));

CREATE TABLE IF NOT EXISTS brain_personas (
    entry_id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    division TEXT,
    identity TEXT,
    critical_rules TEXT,
    communication_style TEXT,
    full_body TEXT,
    domain_keywords TEXT[] NOT NULL DEFAULT '{{}}',
    embedding vector({CONFIG.embedding_dim}),
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS brain_personas_keywords
    ON brain_personas USING gin (domain_keywords);

CREATE TABLE IF NOT EXISTS brain_edges (
    edge_id BIGSERIAL PRIMARY KEY,
    source_table TEXT NOT NULL,
    source_entry_id BIGINT NOT NULL,
    target_table TEXT NOT NULL,
    target_entry_id BIGINT NOT NULL,
    relation_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    UNIQUE (source_table, source_entry_id, target_table, target_entry_id, relation_type)
);
CREATE INDEX IF NOT EXISTS brain_edges_source ON brain_edges (source_table, source_entry_id);
"""


async def init_schema() -> None:
    async with acquire() as conn:
        await conn.execute(SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Generic CRUD helpers
# ---------------------------------------------------------------------------

async def upsert_code(
    repo_path: str,
    file_path: str,
    language: str,
    symbol_name: str,
    symbol_type: str,
    content: str,
    summary: str,
    embedding: Sequence[float],
) -> int:
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO brain_code (repo_path, file_path, language, symbol_name,
                                    symbol_type, content, summary, embedding)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            ON CONFLICT (repo_path, file_path, symbol_name)
            DO UPDATE SET content=EXCLUDED.content,
                          summary=EXCLUDED.summary,
                          embedding=EXCLUDED.embedding,
                          language=EXCLUDED.language,
                          symbol_type=EXCLUDED.symbol_type
            RETURNING entry_id
            """,
            repo_path, file_path, language, symbol_name, symbol_type,
            content, summary, list(embedding),
        )
        return int(row["entry_id"])


async def insert_summary(title: str, content: str, keywords: Sequence[str], embedding: Sequence[float]) -> int:
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO brain_summary (title, content, keywords, embedding)
            VALUES ($1,$2,$3,$4) RETURNING entry_id
            """,
            title, content, list(keywords), list(embedding),
        )
        return int(row["entry_id"])


async def insert_experience(
    context_type: str,
    problem: str,
    solution: str,
    outcome: str,
    embedding: Sequence[float],
) -> int:
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO brain_experience (context_type, problem, solution, outcome, embedding)
            VALUES ($1,$2,$3,$4,$5) RETURNING entry_id
            """,
            context_type, problem, solution, outcome, list(embedding),
        )
        return int(row["entry_id"])


async def upsert_skill(
    name: str, description: str, body: str,
    source_repo: str, source_url: str, embedding: Sequence[float],
) -> int:
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO brain_skills (name, description, body, source_repo, source_url, embedding, updated_at)
            VALUES ($1,$2,$3,$4,$5,$6, now())
            ON CONFLICT (name) DO UPDATE SET
                description=EXCLUDED.description,
                body=EXCLUDED.body,
                source_repo=EXCLUDED.source_repo,
                source_url=EXCLUDED.source_url,
                embedding=EXCLUDED.embedding,
                updated_at=now()
            RETURNING entry_id
            """,
            name, description, body, source_repo, source_url, list(embedding),
        )
        return int(row["entry_id"])


async def upsert_persona(
    name: str, description: str, division: str,
    identity: str, critical_rules: str, communication_style: str,
    full_body: str, domain_keywords: Sequence[str], embedding: Sequence[float],
) -> int:
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO brain_personas (name, description, division, identity,
                critical_rules, communication_style, full_body, domain_keywords,
                embedding, updated_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9, now())
            ON CONFLICT (name) DO UPDATE SET
                description=EXCLUDED.description,
                division=EXCLUDED.division,
                identity=EXCLUDED.identity,
                critical_rules=EXCLUDED.critical_rules,
                communication_style=EXCLUDED.communication_style,
                full_body=EXCLUDED.full_body,
                domain_keywords=EXCLUDED.domain_keywords,
                embedding=EXCLUDED.embedding,
                updated_at=now()
            RETURNING entry_id
            """,
            name, description, division, identity, critical_rules,
            communication_style, full_body, list(domain_keywords), list(embedding),
        )
        return int(row["entry_id"])


async def insert_edge(
    source_table: str, source_id: int,
    target_table: str, target_id: int,
    relation_type: str, confidence: float = 1.0,
) -> None:
    async with acquire() as conn:
        await conn.execute(
            """
            INSERT INTO brain_edges (source_table, source_entry_id, target_table,
                                     target_entry_id, relation_type, confidence)
            VALUES ($1,$2,$3,$4,$5,$6)
            ON CONFLICT DO NOTHING
            """,
            source_table, source_id, target_table, target_id, relation_type, confidence,
        )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

VECTOR_TABLES = {
    "brain_code":       ("entry_id", "symbol_name", "content", 1.0),
    "brain_summary":    ("entry_id", "title", "content", 0.9),
    "brain_experience": ("entry_id", "problem", "solution", 0.8),
}


async def vector_search(table: str, embedding: Sequence[float], top_k: int) -> list[dict[str, Any]]:
    title_col, _, body_col, weight = VECTOR_TABLES[table]
    async with acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT entry_id, {title_col} AS title, {body_col} AS body,
                   1 - (embedding <=> $1) AS score
            FROM {table}
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            list(embedding), top_k,
        )
    return [
        {
            "table": table,
            "entry_id": int(r["entry_id"]),
            "title": r["title"],
            "body": r["body"],
            "score": float(r["score"]) * weight,
            "source": "vector",
        }
        for r in rows
    ]


async def fts_search(table: str, query: str, top_k: int) -> list[dict[str, Any]]:
    title_col, _, body_col, _ = VECTOR_TABLES[table]
    async with acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT entry_id, {title_col} AS title, {body_col} AS body,
                   ts_rank(to_tsvector('english', {title_col} || ' ' || {body_col}),
                           plainto_tsquery('english', $1)) AS score
            FROM {table}
            WHERE to_tsvector('english', {title_col} || ' ' || {body_col})
                  @@ plainto_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT $2
            """,
            query, top_k,
        )
    return [
        {
            "table": table,
            "entry_id": int(r["entry_id"]),
            "title": r["title"],
            "body": r["body"],
            "score": float(r["score"]) * 0.6,
            "source": "fts",
        }
        for r in rows
    ]


async def skill_vector_search(embedding: Sequence[float], top_k: int) -> list[dict[str, Any]]:
    async with acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT entry_id, name, description, body,
                   1 - (embedding <=> $1) AS score
            FROM brain_skills
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            list(embedding), top_k,
        )
    return [
        {
            "table": "brain_skills",
            "entry_id": int(r["entry_id"]),
            "name": r["name"],
            "description": r["description"],
            "body": r["body"],
            "score": float(r["score"]),
        }
        for r in rows
    ]


async def persona_keyword_search(keywords: Sequence[str], top_k: int) -> list[dict[str, Any]]:
    if not keywords:
        return []
    async with acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT entry_id, name, description, division, identity, critical_rules,
                   communication_style, domain_keywords,
                   cardinality(ARRAY(SELECT unnest(domain_keywords)
                                     INTERSECT SELECT unnest($1::text[]))) AS overlap
            FROM brain_personas
            WHERE domain_keywords && $1::text[]
            ORDER BY overlap DESC
            LIMIT $2
            """,
            list(keywords), top_k,
        )
    return [_persona_row(r, source="keyword") for r in rows]


async def persona_vector_search(embedding: Sequence[float], top_k: int) -> list[dict[str, Any]]:
    async with acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT entry_id, name, description, division, identity, critical_rules,
                   communication_style, domain_keywords,
                   1 - (embedding <=> $1) AS score
            FROM brain_personas
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            list(embedding), top_k,
        )
    return [_persona_row(r, source="vector") for r in rows]


def _persona_row(r: asyncpg.Record, source: str) -> dict[str, Any]:
    return {
        "entry_id": int(r["entry_id"]),
        "name": r["name"],
        "description": r["description"],
        "division": r["division"],
        "identity": r["identity"],
        "critical_rules": r["critical_rules"],
        "communication_style": r["communication_style"],
        "domain_keywords": list(r["domain_keywords"] or []),
        "score": float(r["score"]) if "score" in r.keys() else float(r.get("overlap", 0)),
        "source": source,
    }


async def get_neighbors(table: str, entry_id: int, max_depth: int = 1) -> list[dict[str, Any]]:
    async with acquire() as conn:
        rows = await conn.fetch(
            """
            WITH RECURSIVE walk(source_table, source_entry_id, target_table,
                                target_entry_id, relation_type, depth) AS (
                SELECT source_table, source_entry_id, target_table, target_entry_id,
                       relation_type, 1
                FROM brain_edges
                WHERE source_table = $1 AND source_entry_id = $2
                UNION ALL
                SELECT e.source_table, e.source_entry_id, e.target_table,
                       e.target_entry_id, e.relation_type, w.depth + 1
                FROM brain_edges e
                JOIN walk w ON e.source_table = w.target_table
                           AND e.source_entry_id = w.target_entry_id
                WHERE w.depth < $3
            )
            SELECT DISTINCT target_table AS table, target_entry_id AS entry_id, relation_type, depth
            FROM walk
            """,
            table, entry_id, max_depth,
        )
    return [dict(r) for r in rows]


async def touch_access(table: str, entry_ids: Iterable[int]) -> None:
    ids = list(entry_ids)
    if not ids:
        return
    async with acquire() as conn:
        await conn.execute(
            f"""
            UPDATE {table}
            SET access_count = access_count + 1, last_accessed_at = now()
            WHERE entry_id = ANY($1::bigint[])
            """,
            ids,
        )


async def hydrate(table: str, entry_id: int) -> dict[str, Any] | None:
    async with acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT * FROM {table} WHERE entry_id = $1", entry_id
        )
    return dict(row) if row else None


async def stats() -> dict[str, int]:
    counts: dict[str, int] = {}
    tables = ["brain_code", "brain_summary", "brain_experience",
              "brain_skills", "brain_personas", "brain_edges"]
    async with acquire() as conn:
        for t in tables:
            counts[t] = int(await conn.fetchval(f"SELECT count(*) FROM {t}"))
    return counts
