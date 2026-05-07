"""ASGI middleware that injects SSE heartbeats during silent streaming periods.

Cloudflare's edge proxy will close connections that go >100s without bytes,
producing a 524. Reasoning models (DeepSeek V4-Pro, etc.) routinely sit in
their thinking phase that long before emitting the first token.

This middleware wraps streaming responses on configured POST paths and emits
`: heartbeat\\n\\n` SSE comments every BRAIN_CF_HEARTBEAT_SECONDS during gaps.
SSE parsers ignore comment lines, so the keepalive is invisible to clients.

Mount once, in front of LiteLLM's app:
    from cf_streaming_middleware import CFStreamingMiddleware
    app.add_middleware(CFStreamingMiddleware)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from brain_config import CONFIG

logger = logging.getLogger(__name__)

Scope = dict
Message = dict
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


HEARTBEAT_BYTES = b": heartbeat\n\n"


class CFStreamingMiddleware:
    def __init__(self, app: Callable) -> None:
        self.app = app
        self.heartbeat_interval = CONFIG.cf_heartbeat_seconds
        self.streaming_paths = set(CONFIG.cf_streaming_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._should_intercept(scope):
            await self.app(scope, receive, send)
            return

        send_lock = asyncio.Lock()
        last_send = asyncio.get_running_loop().time()
        is_streaming = False
        finished = False

        async def safe_send(message: Message) -> None:
            nonlocal last_send, is_streaming
            async with send_lock:
                last_send = asyncio.get_running_loop().time()
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers") or [])
                    ctype = headers.get(b"content-type", b"")
                    if b"text/event-stream" in ctype:
                        is_streaming = True
                await send(message)

        async def heartbeat_loop() -> None:
            try:
                while not finished:
                    await asyncio.sleep(self.heartbeat_interval / 2)
                    if not is_streaming or finished:
                        continue
                    now = asyncio.get_running_loop().time()
                    if now - last_send < self.heartbeat_interval:
                        continue
                    async with send_lock:
                        if finished:
                            return
                        try:
                            await send({
                                "type": "http.response.body",
                                "body": HEARTBEAT_BYTES,
                                "more_body": True,
                            })
                        except Exception as exc:
                            logger.debug("heartbeat send failed: %s", exc)
                            return
            except asyncio.CancelledError:
                pass

        beat_task = asyncio.create_task(heartbeat_loop(), name="cf-heartbeat")
        try:
            await self.app(scope, receive, safe_send)
        finally:
            finished = True
            beat_task.cancel()
            try:
                await beat_task
            except (asyncio.CancelledError, Exception):
                pass

    def _should_intercept(self, scope: Scope) -> bool:
        if scope.get("type") != "http":
            return False
        if scope.get("method") != "POST":
            return False
        return scope.get("path") in self.streaming_paths
