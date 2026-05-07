"""Capture loop: store valuable model outputs as code chunks or summaries."""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable

import brain_db as db
from brain_config import CONFIG
from brain_embedder import get_embedder

logger = logging.getLogger(__name__)


_CODE_BLOCK = re.compile(r"```(?P<lang>[\w+-]*)\n(?P<body>.*?)```", re.DOTALL)
_SYMBOL_GUESS = re.compile(
    r"^\s*(?:def|class|function|fn|func|interface|struct|type)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


async def capture_if_valuable(user_message: str, assistant_text: str) -> None:
    try:
        for block in _extract_code_blocks(assistant_text):
            await _store_code_chunk(user_message, block)
        if _has_summary_marker(assistant_text):
            await _store_summary(user_message, assistant_text)
    except Exception as exc:
        logger.warning("capture failed: %s", exc)


def _extract_code_blocks(text: str) -> Iterable[dict[str, str]]:
    for m in _CODE_BLOCK.finditer(text):
        body = m.group("body").strip()
        if body.count("\n") + 1 < CONFIG.capture_min_code_lines:
            continue
        yield {"language": (m.group("lang") or "").lower() or "text", "body": body}


def _has_summary_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in CONFIG.capture_summary_markers)


async def _store_code_chunk(user_message: str, block: dict[str, str]) -> None:
    body = block["body"]
    sym_match = _SYMBOL_GUESS.search(body)
    symbol_name = sym_match.group(1) if sym_match else _short_id(body)
    embedder = get_embedder()
    embedding = await embedder.embed(body[:6000])
    summary = (user_message[:200] + "...") if len(user_message) > 200 else user_message
    await db.upsert_code(
        repo_path="session://capture",
        file_path=f"capture/{symbol_name}.{block['language'] or 'txt'}",
        language=block["language"],
        symbol_name=symbol_name,
        symbol_type="captured",
        content=body,
        summary=summary,
        embedding=embedding,
    )


async def _store_summary(user_message: str, assistant_text: str) -> None:
    title = (user_message.splitlines() or ["Summary"])[0][:120]
    embedder = get_embedder()
    embedding = await embedder.embed(assistant_text[:6000])
    keywords = sorted({
        w for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", assistant_text.lower())
    })[:32]
    await db.insert_summary(title, assistant_text, keywords, embedding)


def _short_id(text: str) -> str:
    import hashlib
    return "anon_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
