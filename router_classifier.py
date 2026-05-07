"""Pre-router model override hook.

Runs BEFORE brain_hook in the LiteLLM callback chain. Detects natural-language
overrides in the user's last message and rewrites `data["model"]` so the
gateway dispatches to the requested backend.

Per README §10: only explicit phrases trigger overrides — keyword auto-
detection (frontend/animation/modal/...) was deliberately removed because it
silently burned Moonshot quota on non-design tasks.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

try:
    from litellm.integrations.custom_logger import CustomLogger
except Exception:
    class CustomLogger:  # type: ignore[no-redef]
        async def async_pre_call_hook(self, *a, **k): ...

logger = logging.getLogger(__name__)


_OVERRIDES = [
    (re.compile(r"\buse\s+pro\b", re.IGNORECASE), "deepseek-v4-pro"),
    (re.compile(r"\buse\s+kimi\b", re.IGNORECASE), "kimi-k2.6"),
    (re.compile(r"\buse\s+flash\b", re.IGNORECASE), "deepseek-v4-flash"),
    (re.compile(r"\buse\s+grok\b", re.IGNORECASE), "grok-4.3"),
]


class RouterClassifier(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[dict]:
        try:
            if call_type not in {"completion", "acompletion"}:
                return data
            text = self._last_user_text(data.get("messages") or [])
            if not text:
                return data
            for pattern, target in _OVERRIDES:
                if pattern.search(text):
                    original = data.get("model")
                    data["model"] = target
                    logger.info("router override: %s -> %s", original, target)
                    break
        except Exception as exc:
            logger.warning("router_classifier failed: %s", exc)
        return data

    @staticmethod
    def _last_user_text(messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
        return ""


proxy_handler_instance = RouterClassifier()
