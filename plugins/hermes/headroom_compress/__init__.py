"""Compress Hermes tool results locally without changing provider authentication."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)


def _compress_request(*, request: dict[str, Any], model: str = "", **_: Any) -> dict | None:
    """Densify only tool-result text in the outgoing request, never session history."""
    from headroom import densify

    field = "input" if "input" in request else "messages"
    items = request.get(field)
    if not isinstance(items, list):
        return None

    updated = deepcopy(items)
    tokens_saved = 0

    def compress_text(text: Any) -> Any:
        nonlocal tokens_saved
        if not isinstance(text, str):
            return text
        result = densify(
            [{"role": "tool", "content": text}],
            model=model or request.get("model") or "gpt-4o",
            protect_recent=0,
            compress_system_messages=False,
        )
        content = result.messages[0]["content"]
        if (
            result.lossless is True
            and result.tokens_saved > 0
            and isinstance(content, str)
            and content != text
        ):
            tokens_saved += result.tokens_saved
            return content
        return text

    try:
        for item in updated:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call_output" and isinstance(item.get("output"), str):
                item["output"] = compress_text(item["output"])
            elif item.get("role") == "tool" and isinstance(item.get("content"), str):
                item["content"] = compress_text(item["content"])
            elif item.get("role") == "user" and isinstance(item.get("content"), list):
                for block in item["content"]:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    if block.get("is_error"):
                        continue
                    # Mixed text/image results stay intact; do not cross block boundaries.
                    if isinstance(block.get("content"), str):
                        block["content"] = compress_text(block["content"])
        if tokens_saved:
            logger.info("Headroom saved %d tool-result tokens", tokens_saved)
            return {"request": {**request, field: updated}, "source": "headroom_compress"}
    except Exception as exc:
        # Do not log tool content or exception messages, which can contain secrets.
        logger.warning("Headroom compression skipped (%s)", type(exc).__name__)
    return None


def register(ctx: Any) -> None:
    """Register supported Hermes request middleware; no proxy or token exchange needed."""
    ctx.register_middleware("llm_request", _compress_request)
