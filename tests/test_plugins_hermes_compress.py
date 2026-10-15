"""Hermes request middleware preserves native provider requests and history."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN = Path(__file__).resolve().parents[1] / "plugins/hermes/headroom_compress/__init__.py"


@pytest.fixture
def plugin():
    spec = importlib.util.spec_from_file_location("hermes_headroom_compress", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("field", "item", "key"),
    [
        ("messages", {"role": "tool", "tool_call_id": "call1", "content": "large"}, "content"),
        (
            "input",
            {"type": "function_call_output", "call_id": "call1", "output": "large"},
            "output",
        ),
        (
            "messages",
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call1",
                        "content": "large",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            "anthropic",
        ),
    ],
)
def test_compresses_only_result_text(plugin, monkeypatch, field, item, key):
    import headroom

    seen = []

    def densify(messages, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(
            messages=[{"role": "tool", "content": "small"}], lossless=True, tokens_saved=10
        )

    monkeypatch.setattr(headroom, "densify", densify)
    user = {"role": "user", "content": "Keep my question exactly"}
    request = {
        "model": "gpt-5.4",
        field: [user, item, {"role": "tool"}, {"type": "function_call_output"}],
        "stream": True,
        "extra_headers": {"x-initiator": "agent"},
        "tools": [{"name": "terminal"}],
        "instructions": "Keep system instructions",
    }
    original = deepcopy(request)
    result = plugin._compress_request(request=request, model="gpt-5.4", provider="copilot")
    expected = deepcopy(original)
    if key == "anthropic":
        expected[field][1]["content"][0]["content"] = "small"
    else:
        expected[field][1][key] = "small"
    assert result["request"] == expected
    assert request == original
    assert seen == [{"model": "gpt-5.4", "protect_recent": 0, "compress_system_messages": False}]


@pytest.mark.parametrize("lossless,saved", [(False, 10), (None, 10), (True, 0), (True, -1)])
def test_rejects_unverified_or_inflated_result(plugin, monkeypatch, lossless, saved):
    import headroom

    monkeypatch.setattr(
        headroom,
        "densify",
        lambda *a, **kw: SimpleNamespace(
            messages=[{"content": "changed"}], lossless=lossless, tokens_saved=saved
        ),
    )
    request = {"messages": [{"role": "tool", "content": "original"}]}
    assert plugin._compress_request(request=request) is None
    assert request["messages"][0]["content"] == "original"


def test_errors_pass_through_without_content_in_logs(plugin, monkeypatch, caplog):
    import headroom

    def fail(*args, **kwargs):
        raise ValueError("private tool output")

    monkeypatch.setattr(headroom, "densify", fail)
    request = {"messages": [{"role": "tool", "content": "original"}]}
    assert plugin._compress_request(request=request) is None
    assert "private tool output" not in caplog.text
    assert "ValueError" in caplog.text


@pytest.mark.parametrize("outcome", ["success", "failure", "unchanged"])
def test_logs_only_committed_request_savings(plugin, monkeypatch, caplog, outcome):
    import logging

    import headroom

    calls = 0

    def densify(messages, **kwargs):
        nonlocal calls
        calls += 1
        if outcome == "failure" and calls == 2:
            raise ValueError("second result failed")
        content = messages[0]["content"] if outcome == "unchanged" else "small"
        return SimpleNamespace(
            messages=[{"content": content}], lossless=True, tokens_saved=10 * calls
        )

    monkeypatch.setattr(headroom, "densify", densify)
    request = {
        "messages": [{"role": "tool", "content": "first"}, {"role": "tool", "content": "second"}]
    }
    original = deepcopy(request)
    with caplog.at_level(logging.INFO, logger=plugin.logger.name):
        result = plugin._compress_request(request=request)

    savings = [r.getMessage() for r in caplog.records if "Headroom saved" in r.getMessage()]
    assert request == original
    if outcome == "success":
        assert result["request"]["messages"] == [{"role": "tool", "content": "small"}] * 2
        assert savings == ["Headroom saved 30 tool-result tokens"]
    else:
        assert result is None
        assert savings == []


def test_preserves_non_tool_and_multimodal_content(plugin, monkeypatch):
    import headroom

    def fail(*args, **kwargs):
        pytest.fail("Non-tool, error, and multimodal content must not be compressed")

    monkeypatch.setattr(headroom, "densify", fail)
    request = {
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "assistant", "content": "answer", "tool_calls": [{"id": "a"}]},
            {
                "role": "tool",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,a"}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "question"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "b",
                        "is_error": True,
                        "content": "error",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "c",
                        "content": [{"type": "image", "source": {}}],
                    },
                ],
            },
        ]
    }
    original = deepcopy(request)
    assert plugin._compress_request(request=request) is None
    assert request == original
    assert plugin._compress_request(request={"input": "hello"}) is None


def test_registers_native_middleware(plugin):
    registered = []
    plugin.register(SimpleNamespace(register_middleware=lambda *args: registered.append(args)))
    assert registered == [("llm_request", plugin._compress_request)]


def test_real_lossless_compression(plugin):
    import json

    from headroom.transforms.compaction_codec import expand_compacted

    text = json.dumps(
        [{"id": i, "status": "healthy", "service": "worker", "region": "east"} for i in range(100)],
        indent=2,
    )
    request = {"input": [{"type": "function_call_output", "call_id": "a", "output": text}]}
    result = plugin._compress_request(request=request, model="gpt-4o")
    assert result is not None
    compressed = result["request"]["input"][0]["output"]
    assert len(compressed) < len(text)
    assert expand_compacted(compressed) == json.loads(text)
    assert request["input"][0]["output"] == text
    assert plugin._compress_request(request=request, model="gpt-4o") == result
