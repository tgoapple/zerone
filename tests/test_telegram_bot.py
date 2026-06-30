from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from telegram_bot import _build_agent, _chunk_text, _help_text, _status_text, _summarize_telegram_reply


def test_chunk_text_keeps_short_text():
    text = "hello"
    assert _chunk_text(text, limit=10) == ["hello"]


def test_chunk_text_splits_long_text():
    text = ("alpha beta gamma delta " * 20).strip()
    chunks = _chunk_text(text, limit=80)
    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert "alpha" in chunks[0]


def test_help_text_mentions_status():
    text = _help_text("ZEROne")
    assert "ZEROne" in text
    assert "/status" in text


def test_status_text_includes_provider_and_workspace(tmp_path: Path):
    args = SimpleNamespace(
        name="ZEROne",
        provider="deepseek",
        model="deepseek-v4-flash",
        workspace=str(tmp_path),
    )
    agent = _build_agent(args)
    text = _status_text(agent)
    assert "provider: deepseek" in text
    assert "workspace:" in text
    assert str(tmp_path) in text


def test_summarize_telegram_reply_uses_tool_steps():
    reply = {
        "content": "<html>ignored</html>",
        "meta": {
            "agent_steps": [
                {"tool": "open_target", "args": {"target": "mip-framework/index.html"}},
            ]
        },
    }
    assert _summarize_telegram_reply(reply) == "Opened mip-framework/index.html on this Mac."


def test_summarize_telegram_reply_hides_raw_html():
    reply = {"content": "```html\n<!DOCTYPE html><html></html>\n```", "meta": {}}
    assert "raw HTML" in _summarize_telegram_reply(reply)
