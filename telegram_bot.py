#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from urllib import error, request as urllib_request

from zerone import Config, ZEROne, _extract_html, _looks_like_code_dump

TELEGRAM_TEXT_LIMIT = 3900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZEROne Telegram bridge")
    parser.add_argument("--name", default="ZEROne")
    parser.add_argument("--provider", default=os.environ.get("MIP_PROVIDER", "deepseek"))
    parser.add_argument("--model", default=os.environ.get("MIP_MODEL"))
    parser.add_argument("--workspace", default=os.environ.get("MIP_WORKSPACE_ROOT"))
    parser.add_argument("--poll-timeout", type=int, default=60)
    return parser.parse_args()


def call_telegram(method: str, token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{token}/{method}"
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib_request.Request(url, data=body, headers=headers, method="POST")
    import sys as _sys
    try:
        with urllib_request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        _sys.stderr.write(f"call_telegram HTTP {exc.code} for {method}: {exc.reason}\n")
        _sys.stderr.flush()
        if exc.code == 409:
            raise  # let the caller retry
        raise RuntimeError(f"HTTP {exc.code}: {exc.reason}")
    if not data.get("ok", True):
        description = data.get("description", "Telegram request failed.")
        raise RuntimeError(description)
    return data


def _chunk_text(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        split_at = remaining.rfind("\n\n", 0, limit)
        if split_at < 0:
            split_at = remaining.rfind("\n", 0, limit)
        if split_at < 0:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at < 0:
            split_at = limit

        piece = remaining[:split_at].strip()
        if not piece:
            piece = remaining[:limit].strip()
            split_at = limit

        chunks.append(piece)
        remaining = remaining[split_at:].strip()

    return chunks or [text]


def send_chat_action(token: str, chat_id: int, action: str = "typing") -> None:
    call_telegram("sendChatAction", token, {"chat_id": chat_id, "action": action})


def send_message(token: str, chat_id: int, text: str) -> None:
    for chunk in _chunk_text(text):
        call_telegram(
            "sendMessage",
            token,
            {
                "chat_id": chat_id,
                "text": chunk,
                "disable_web_page_preview": True,
            },
        )


def _summarize_telegram_reply(reply: dict[str, Any]) -> str:
    content = str(reply.get("content", "")).strip()
    meta = reply.get("meta", {}) or {}
    steps = meta.get("agent_steps", []) or []
    if steps:
        last = steps[-1]
        tool = str(last.get("tool", "")).strip()
        args = last.get("args", {}) or {}
        if tool == "open_target":
            return f"Opened {args.get('target', '')} on this Mac."
        if tool == "build_landing_page":
            path = str(args.get("path", "")).strip()
            mode = str(args.get("mode", "create"))
            if mode == "improve":
                return f"Improved and opened {path} on this Mac."
            return f"Created and opened {path} on this Mac."
        if tool == "write_file":
            return f"Written to {args.get('path', '')}."
    if _extract_html(content) is not None:
        return "I wrote the page, but I’m not going to dump the raw HTML into Telegram."
    if _looks_like_code_dump(content):
        return "I handled that as code, but I’m keeping the raw code out of Telegram."
    return content


def _typing_pulse(token: str, chat_id: int, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            send_chat_action(token, chat_id, "typing")
        except Exception:
            pass
        stop.wait(4)


def _help_text(name: str) -> str:
    return (
        f"{name}\n"
        "Telegram companion + operator\n\n"
        "Send a message naturally and I'll respond in the same ZEROne runtime you use in the CLI.\n\n"
        "Useful commands:\n"
        "/help - show this help\n"
        "/status - show provider and workspace"
    )


def _status_text(agent: ZEROne) -> str:
    workspace = str(agent._workspace_root)
    model = getattr(agent.provider, "model", None) or "default"
    return (
        f"{agent.name} status\n"
        f"provider: {agent.provider.name}\n"
        f"model: {model}\n"
        f"workspace: {workspace}"
    )


def _build_agent(args: argparse.Namespace) -> ZEROne:
    return ZEROne(
        Config(
            assistant_name=args.name,
            provider_name=args.provider,
            visual_provider_name=os.environ.get("MIP_VISUAL_PROVIDER", "codex"),
            model=args.model,
            workspace_root=args.workspace,
            data_dir=Path(__file__).resolve().parent / "data",
        )
    )


def main() -> int:
    args = parse_args()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("TELEGRAM_BOT_TOKEN is not configured.")
        return 1

    agent = _build_agent(args)
    offset = 0

    print(f"{agent.name} Telegram bridge running.")
    print(f"provider={agent.provider.name} model={getattr(agent.provider, 'model', None) or 'default'}")
    print(f"workspace={agent._workspace_root}")

    while True:
        try:
            updates = call_telegram(
                "getUpdates",
                token,
                {"timeout": args.poll_timeout, "offset": offset},
            )
            for item in updates.get("result", []):
                offset = max(offset, item["update_id"] + 1)
                message = item.get("message") or item.get("edited_message")
                if not message or "text" not in message:
                    continue
                if message.get("from", {}).get("is_bot"):
                    continue

                chat_id = message["chat"]["id"]
                session_id = f"telegram-{chat_id}"
                user_text = message["text"].strip()
                if not user_text:
                    continue

                print(f"telegram<{chat_id}> {user_text[:100]}", flush=True)

                if user_text in {"/start", "/help"}:
                    send_message(token, chat_id, _help_text(agent.name))
                    continue
                if user_text == "/status":
                    send_message(token, chat_id, _status_text(agent))
                    continue

                stop = threading.Event()
                pulse = threading.Thread(target=_typing_pulse, args=(token, chat_id, stop), daemon=True)
                pulse.start()
                try:
                    reply = agent.reply(
                        session_id,
                        user_text,
                        metadata={"surface": "telegram", "chat_id": chat_id},
                    )
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    send_message(token, chat_id, f"Error: {exc}")
                else:
                    summary = _summarize_telegram_reply(reply)
                    if summary:
                        try:
                            send_message(token, chat_id, summary)
                        except Exception as exc2:
                            import traceback
                            print(f"send_message failed: {exc2}", flush=True)
                            traceback.print_exc()
                    else:
                        content = str(reply.get("content", "")).strip()
                        if content:
                            try:
                                send_message(token, chat_id, content)
                            except Exception as exc2:
                                import traceback
                                print(f"send_message (raw) failed: {exc2}", flush=True)
                                traceback.print_exc()
                finally:
                    stop.set()
                    pulse.join(timeout=0.2)
        except error.HTTPError as exc:
            if exc.code == 409:
                # Another long-poll is active — wait and retry
                import sys; print("telegram> 409 conflict — retrying", flush=True)
                time.sleep(3)
                continue
            import sys; print(f"telegram> HTTP {exc.code}: {exc.reason}", flush=True)
            time.sleep(5)
        except error.URLError as exc:
            print(f"telegram> network error: {exc.reason}")
            time.sleep(5)
        except Exception as exc:
            print(f"telegram> error: {exc}")
            time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
