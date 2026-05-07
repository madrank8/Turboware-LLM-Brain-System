"""Formats a RetrievalResult into the system message that Brain prepends.

Layout (per design decision §12 in README):
  1. Persona identity prefix          (~600 token budget)
  2. Code/summary/experience contexts
  3. Skill methodology blocks         (dedicated slots)

We don't enforce a hard token cap here — the gateway is responsible for
truncation if the combined messages exceed model context.
"""
from __future__ import annotations

from typing import Iterable

from brain_retriever import RetrievalResult


_HEADER = "# Brain Context\n\nThe following context was retrieved from your team's shared memory.\n"


def format_system_message(result: RetrievalResult) -> str:
    sections: list[str] = []
    if result.personas:
        sections.append(_format_personas(result.personas))
    if result.contexts:
        sections.append(_format_contexts(result.contexts))
    if result.skills:
        sections.append(_format_skills(result.skills))
    if not sections:
        return ""
    return _HEADER + "\n\n".join(sections)


def _format_personas(personas: Iterable[dict]) -> str:
    blocks: list[str] = []
    for p in personas:
        identity = (p.get("identity") or "").strip()
        rules = (p.get("critical_rules") or "").strip()
        style = (p.get("communication_style") or "").strip()
        block = f"## Identity: {p['name']}"
        if p.get("division"):
            block += f"  ({p['division']})"
        if identity:
            block += f"\n\n{identity}"
        if rules:
            block += f"\n\n### Critical rules\n{rules}"
        if style:
            block += f"\n\n### Communication style\n{style}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _format_contexts(contexts: Iterable[dict]) -> str:
    out = ["## Relevant code & history"]
    for c in contexts:
        title = c.get("title") or c.get("table")
        body = (c.get("body") or "").strip()
        source = c.get("source", "")
        out.append(f"### {title}  _({c['table']}, {source})_\n```\n{body}\n```")
    return "\n\n".join(out)


def _format_skills(skills: Iterable[dict]) -> str:
    out = ["## Applicable skills"]
    for s in skills:
        name = s.get("name", "skill")
        desc = (s.get("description") or "").strip()
        body = (s.get("body") or "").strip()
        out.append(f"### {name}\n{desc}\n\n{body}")
    return "\n\n".join(out)
