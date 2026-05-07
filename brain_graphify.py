"""Wrapper around the Graphify CLI for AST-quality code ingestion.

Graphify (https://github.com/safishamsi/graphify) parses 25 languages with
tree-sitter and emits a JSON knowledge graph. We map its nodes/edges into
brain_code rows and brain_edges relationships.

Expected graphify JSON shape (subset we rely on):
    {
      "nodes": [
        {"id": "...", "type": "function|class|...",
         "name": "...", "file": "rel/path", "language": "python",
         "code": "...", "doc": "..." }
      ],
      "edges": [
        {"source": "...", "target": "...", "relation": "calls|contains|imports"}
      ]
    }
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import brain_db as db
from brain_config import CONFIG
from brain_embedder import get_embedder

logger = logging.getLogger(__name__)


async def graphify_repo(repo_path: str) -> dict[str, int]:
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        raise ValueError(f"not a directory: {repo}")
    graph = await _run_graphify(repo)
    embedder = get_embedder()

    id_to_db_id: dict[str, tuple[str, int]] = {}
    nodes_inserted = 0
    for node in graph.get("nodes", []):
        if node.get("type") not in {"function", "class", "method", "struct", "interface"}:
            continue
        body = node.get("code") or ""
        if not body.strip():
            continue
        embedding = await embedder.embed(body[:6000])
        entry_id = await db.upsert_code(
            repo_path=str(repo),
            file_path=node.get("file", ""),
            language=node.get("language", ""),
            symbol_name=node.get("name", ""),
            symbol_type=node.get("type", ""),
            content=body,
            summary=(node.get("doc") or "")[:500],
            embedding=embedding,
        )
        id_to_db_id[node["id"]] = ("brain_code", entry_id)
        nodes_inserted += 1

        if (doc := node.get("doc")):
            doc_embedding = await embedder.embed(doc[:4000])
            doc_id = await db.insert_summary(
                title=f"{node.get('name','')} rationale",
                content=doc,
                keywords=[],
                embedding=doc_embedding,
            )
            await db.insert_edge(
                "brain_summary", doc_id, "brain_code", entry_id, "rationale_for"
            )

    edges_inserted = 0
    for edge in graph.get("edges", []):
        src = id_to_db_id.get(edge.get("source", ""))
        tgt = id_to_db_id.get(edge.get("target", ""))
        if not src or not tgt:
            continue
        await db.insert_edge(
            src[0], src[1], tgt[0], tgt[1],
            edge.get("relation", "related"),
            float(edge.get("confidence", 1.0)),
        )
        edges_inserted += 1

    return {"nodes": nodes_inserted, "edges": edges_inserted}


async def _run_graphify(repo: Path) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        proc = await asyncio.create_subprocess_exec(
            CONFIG.graphify_binary, "--repo", str(repo), "--out", tmp_path,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"graphify failed ({proc.returncode}): {stderr.decode(errors='replace')}"
            )
        return json.loads(Path(tmp_path).read_text())
    finally:
        Path(tmp_path).unlink(missing_ok=True)
