"""Fetch SKILL.md files from GitHub raw URLs and store them in brain_skills."""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import httpx
import yaml

import brain_db as db
from brain_embedder import get_embedder

logger = logging.getLogger(__name__)


# Curated registry of skills referenced in the README. Each entry is a
# (source_repo, raw_url) pair pointing to a SKILL.md file.
DEFAULT_SKILLS: list[tuple[str, str]] = [
    ("mattpocock/skills", "https://raw.githubusercontent.com/mattpocock/skills/main/diagnose/SKILL.md"),
    ("mattpocock/skills", "https://raw.githubusercontent.com/mattpocock/skills/main/improve-codebase-architecture/SKILL.md"),
    ("mattpocock/skills", "https://raw.githubusercontent.com/mattpocock/skills/main/grill-with-docs/SKILL.md"),
    ("mattpocock/skills", "https://raw.githubusercontent.com/mattpocock/skills/main/zoom-out/SKILL.md"),
    ("affaan-m/everything-claude-code", "https://raw.githubusercontent.com/affaan-m/everything-claude-code/main/skills/mcp-server-patterns/SKILL.md"),
    ("affaan-m/everything-claude-code", "https://raw.githubusercontent.com/affaan-m/everything-claude-code/main/skills/postgres-patterns/SKILL.md"),
    ("affaan-m/everything-claude-code", "https://raw.githubusercontent.com/affaan-m/everything-claude-code/main/skills/api-design/SKILL.md"),
    ("affaan-m/everything-claude-code", "https://raw.githubusercontent.com/affaan-m/everything-claude-code/main/skills/deployment-patterns/SKILL.md"),
    ("affaan-m/everything-claude-code", "https://raw.githubusercontent.com/affaan-m/everything-claude-code/main/skills/cost-aware-llm-pipeline/SKILL.md"),
    ("affaan-m/everything-claude-code", "https://raw.githubusercontent.com/affaan-m/everything-claude-code/main/skills/python-patterns/SKILL.md"),
    ("affaan-m/everything-claude-code", "https://raw.githubusercontent.com/affaan-m/everything-claude-code/main/skills/docker-patterns/SKILL.md"),
    ("affaan-m/everything-claude-code", "https://raw.githubusercontent.com/affaan-m/everything-claude-code/main/skills/security-review/SKILL.md"),
    ("anthropics/skills", "https://raw.githubusercontent.com/anthropics/skills/main/document-skills/docx/SKILL.md"),
    ("anthropics/skills", "https://raw.githubusercontent.com/anthropics/skills/main/document-skills/pdf/SKILL.md"),
    ("anthropics/skills", "https://raw.githubusercontent.com/anthropics/skills/main/document-skills/pptx/SKILL.md"),
    ("anthropics/skills", "https://raw.githubusercontent.com/anthropics/skills/main/document-skills/xlsx/SKILL.md"),
    ("obra/superpowers", "https://raw.githubusercontent.com/obra/superpowers/main/finishing-a-development-branch/SKILL.md"),
    ("obra/superpowers", "https://raw.githubusercontent.com/obra/superpowers/main/systematic-debugging/SKILL.md"),
    ("obra/superpowers", "https://raw.githubusercontent.com/obra/superpowers/main/using-git-worktrees/SKILL.md"),
    ("obra/superpowers", "https://raw.githubusercontent.com/obra/superpowers/main/subagent-driven-development/SKILL.md"),
    ("obra/superpowers", "https://raw.githubusercontent.com/obra/superpowers/main/test-driven-development/SKILL.md"),
    ("obra/superpowers", "https://raw.githubusercontent.com/obra/superpowers/main/brainstorming/SKILL.md"),
    ("yvgude/lean-ctx", "https://raw.githubusercontent.com/yvgude/lean-ctx/main/SKILL.md"),
    ("forrestchang/andrej-karpathy-skills", "https://raw.githubusercontent.com/forrestchang/andrej-karpathy-skills/main/karpathy-guidelines/SKILL.md"),
]


async def ingest_skills(skills: Iterable[tuple[str, str]] | None = None) -> dict[str, int]:
    skills = list(skills or DEFAULT_SKILLS)
    embedder = get_embedder()
    succeeded = failed = 0
    async with httpx.AsyncClient(timeout=15.0) as http:
        for source_repo, url in skills:
            try:
                resp = await http.get(url, follow_redirects=True)
                resp.raise_for_status()
                parsed = _parse_skill_md(resp.text)
                if parsed is None:
                    failed += 1
                    continue
                embed_text = f"{parsed['name']}\n{parsed['description']}"
                embedding = await embedder.embed(embed_text)
                await db.upsert_skill(
                    name=parsed["name"],
                    description=parsed["description"],
                    body=parsed["body"],
                    source_repo=source_repo,
                    source_url=url,
                    embedding=embedding,
                )
                succeeded += 1
            except Exception as exc:
                logger.warning("skill %s failed: %s", url, exc)
                failed += 1
    return {"succeeded": succeeded, "failed": failed, "total": len(skills)}


def _parse_skill_md(text: str) -> dict[str, Any] | None:
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        front = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None
    name = front.get("name")
    if not name:
        return None
    return {
        "name": str(name),
        "description": str(front.get("description", "")),
        "body": parts[2].strip(),
    }
