"""Fetch and store agent personas from msitarzewski/agency-agents.

For each `.md` file we walk the repo tree via the GitHub Trees API, fetch the
raw content, parse YAML frontmatter + structured sections, extract
domain_keywords, embed (name + description + identity), and upsert.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Optional

import httpx
import yaml

from brain_config import CONFIG
import brain_db as db
from brain_embedder import get_embedder

logger = logging.getLogger(__name__)


_SECTION_PATTERNS = {
    "identity": re.compile(r"##\s*Identity[^\n]*\n(.*?)(?=\n##\s|\Z)", re.DOTALL | re.IGNORECASE),
    "critical_rules": re.compile(r"##\s*Critical Rules[^\n]*\n(.*?)(?=\n##\s|\Z)", re.DOTALL | re.IGNORECASE),
    "communication_style": re.compile(
        r"##\s*Communication Style[^\n]*\n(.*?)(?=\n##\s|\Z)", re.DOTALL | re.IGNORECASE
    ),
}


async def ingest_personas() -> dict[str, int]:
    owner = CONFIG.persona_repo_owner
    repo = CONFIG.persona_repo_name
    ref = CONFIG.persona_repo_ref

    succeeded = failed = 0
    async with httpx.AsyncClient(timeout=20.0) as http:
        tree = await _fetch_tree(http, owner, repo, ref)
        embedder = get_embedder()
        for entry in tree:
            path = entry.get("path", "")
            if not path.endswith(".md") or entry.get("type") != "blob":
                continue
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
            try:
                resp = await http.get(raw_url)
                resp.raise_for_status()
                parsed = _parse_persona(resp.text, path)
                if parsed is None:
                    continue
                embed_text = "\n".join([
                    parsed["name"], parsed["description"], parsed["identity"][:1000]
                ])
                embedding = await embedder.embed(embed_text)
                await db.upsert_persona(
                    name=parsed["name"],
                    description=parsed["description"],
                    division=parsed["division"],
                    identity=parsed["identity"],
                    critical_rules=parsed["critical_rules"],
                    communication_style=parsed["communication_style"],
                    full_body=parsed["full_body"],
                    domain_keywords=parsed["domain_keywords"],
                    embedding=embedding,
                )
                succeeded += 1
            except Exception as exc:
                logger.warning("persona %s failed: %s", path, exc)
                failed += 1
    return {"succeeded": succeeded, "failed": failed}


async def _fetch_tree(http: httpx.AsyncClient, owner: str, repo: str, ref: str) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
    resp = await http.get(url)
    resp.raise_for_status()
    return resp.json().get("tree", [])


def _parse_persona(text: str, path: str) -> Optional[dict[str, Any]]:
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
    body = parts[2].strip()

    division = path.split("/")[0] if "/" in path else "uncategorized"
    sections = {key: _extract_section(body, pat) for key, pat in _SECTION_PATTERNS.items()}
    keywords_seed = " ".join([
        str(name), str(front.get("description", "")),
        sections.get("identity", ""), division,
    ])
    return {
        "name": str(name),
        "description": str(front.get("description", "")),
        "division": division,
        "identity": sections["identity"],
        "critical_rules": sections["critical_rules"],
        "communication_style": sections["communication_style"],
        "full_body": body,
        "domain_keywords": _extract_keywords(keywords_seed),
    }


def _extract_section(body: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(body)
    return match.group(1).strip() if match else ""


_KEYWORD_STOP = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "for", "with",
    "of", "to", "in", "on", "at", "by", "as", "be", "this", "that", "from",
    "into", "their", "they", "you", "your", "we", "our",
}


def _extract_keywords(text: str, limit: int = 24) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok in _KEYWORD_STOP or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return out
