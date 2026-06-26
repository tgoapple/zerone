#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib import error, parse, request as urllib_request

from zerone import ZEROne, Config


def call_telegram(method: str, token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{token}/{method}"
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib_request.Request(url, data=body, headers=headers, method="POST")
    with urllib_request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def send_message(token: str, chat_id: int, text: str) -> None:
    call_telegram("sendMessage", token, {"chat_id": chat_id, "text": text})


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("TELEGRAM_BOT_TOKEN is not configured.")
        return 1

    agent = ZEROne(
        Config(
            provider_name=os.environ.get("MIP_PROVIDER", "deepseek"),
            model=os.environ.get("MIP_MODEL"),
            data_dir=Path(__file__).resolve().parent / "data",
        )
    )

    offset = 0
    print("Telegram bridge running.")
    while True:
        try:
            updates = call_telegram(
                "getUpdates",
                token,
                {"timeout": 60, "offset": offset},
            )
            for item in updates.get("result", []):
                offset = max(offset, item["update_id"] + 1)
                message = item.get("message") or item.get("edited_message")
                if not message or "text" not in message:
                    continue

                chat_id = message["chat"]["id"]
                session_id = f"telegram-{chat_id}"
                user_text = message["text"].strip()
                if not user_text:
                    continue

                try:
                    reply = agent.reply(
                        session_id,
                        user_text,
                        metadata={"surface": "telegram", "chat_id": chat_id},
                    )
                    send_message(token, chat_id, reply["content"])
                except Exception as exc:
                    send_message(token, chat_id, f"Error: {exc}")
        except error.URLError as exc:
            print(f"telegram> network error: {exc.reason}")
            time.sleep(5)
        except Exception as exc:
            print(f"telegram> error: {exc}")
            time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
