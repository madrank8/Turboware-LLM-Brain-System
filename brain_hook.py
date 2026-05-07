"""LiteLLM CustomLogger that wires Brain into every coding-model request.

Pre-call: retrieve context and prepend it as a system message.
Post-call: fire-and-forget capture + learn loops.

Errors anywhere are logged and swallowed — no Brain failure ever blocks
the user request.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

try:
    from litellm.integrations.custom_logger import CustomLogger
except Exception:  # litellm not present in unit-test environments
    class CustomLogger:  # type: ignore[no-redef]
        async def async_pre_call_hook(self, *a, **k): ...
        async def async_post_call_success_hook(self, *a, **k): ...

import brain_db as db
from brain_formatter import format_system_message
from brain_ingest import capture_if_valuable
from brain_learn import learn_if_problem
from brain_retriever import retrieve

logger = logging.getLogger(__name__)


_CODE_BLOCK = re.compile(r"```(?:[\w+-]+)?\n(.*?)```", re.DOTALL)
_CODING_MODEL_TAGS = ("coder", "code", "smart-router", "deepseek", "kimi", "grok", "gpt-")


class BrainHook(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_init(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await db.init_pool()
            self._initialized = True

    # ------------------------------------------------------------------
    # Pre-call: inject retrieved context
    # ------------------------------------------------------------------
    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict | None:
        try:
            await self._ensure_init()
            if not self._is_coding_call(data, call_type):
                return data
            messages = data.get("messages") or []
            user_msg = self._last_user_message(messages)
            if not user_msg:
                return data
            code_blocks = _CODE_BLOCK.findall(user_msg)
            result = await retrieve(user_msg, code_blocks=code_blocks)
            system_text = format_system_message(result)
            if system_text:
                data["messages"] = [{"role": "system", "content": system_text}, *messages]
            logger.debug("brain retrieval %s", result.timings_ms)
        except Exception as exc:
            logger.warning("brain pre-call failed: %s", exc)
        return data

    # ------------------------------------------------------------------
    # Post-call: capture + learn (fire-and-forget)
    # ------------------------------------------------------------------
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: Any,
    ) -> None:
        try:
            await self._ensure_init()
            messages = data.get("messages") or []
            user_msg = self._last_user_message(messages)
            assistant_text = self._extract_assistant_text(response)
            if not user_msg or not assistant_text:
                return
            asyncio.create_task(capture_if_valuable(user_msg, assistant_text))
            asyncio.create_task(learn_if_problem(user_msg, assistant_text))
        except Exception as exc:
            logger.warning("brain post-call failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_coding_call(data: dict, call_type: str) -> bool:
        if call_type not in {"completion", "acompletion"}:
            return False
        model = (data.get("model") or "").lower()
        return any(tag in model for tag in _CODING_MODEL_TAGS)

    @staticmethod
    def _last_user_message(messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                    return "\n".join(p for p in parts if p)
        return ""

    @staticmethod
    def _extract_assistant_text(response: Any) -> str:
        try:
            choices = getattr(response, "choices", None) or response.get("choices", [])
            if not choices:
                return ""
            msg = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
            if msg is None:
                return ""
            if isinstance(msg, dict):
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
            else:
                content = getattr(msg, "content", "") or ""
                reasoning = getattr(msg, "reasoning_content", "") or ""
            return (reasoning + "\n" + content).strip() if reasoning else content
        except Exception:
            return ""


proxy_handler_instance = BrainHook()
