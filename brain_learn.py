"""Learn loop: detect problem-solving exchanges and store them as experiences."""
from __future__ import annotations

import logging

from brain_config import CONFIG
import brain_db as db
from brain_embedder import get_embedder

logger = logging.getLogger(__name__)


_MIN_SOLUTION_CHARS = 200


async def learn_if_problem(user_message: str, assistant_text: str) -> None:
    try:
        if not _has_problem_marker(user_message):
            return
        if len(assistant_text.strip()) < _MIN_SOLUTION_CHARS:
            return
        embedder = get_embedder()
        embedding = await embedder.embed((user_message + "\n" + assistant_text)[:6000])
        await db.insert_experience(
            context_type=_classify(user_message),
            problem=user_message[:4000],
            solution=assistant_text[:8000],
            outcome="captured",
            embedding=embedding,
        )
    except Exception as exc:
        logger.warning("learn failed: %s", exc)


def _has_problem_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in CONFIG.learn_problem_markers)


def _classify(text: str) -> str:
    lower = text.lower()
    if "performance" in lower or "slow" in lower or "latency" in lower:
        return "performance"
    if "architecture" in lower or "design" in lower or "refactor" in lower:
        return "architecture"
    if "error" in lower or "exception" in lower or "traceback" in lower:
        return "error"
    return "bug"
