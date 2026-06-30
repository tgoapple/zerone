#!/usr/bin/env python3
"""ZEROne CLI — sleek terminal interface."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from zerone import ZEROne, Config


# ── Terminal rendering ──────────────────────────────────────

def _use_color() -> bool:
    return os.environ.get("NO_COLOR") is None


def _paint(text: str, code: str) -> str:
    if not _use_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _rule(char: str = "─") -> str:
    return _paint(char * 72, "38;5;240")


def _label(text: str) -> str:
    return _paint(text, "1;38;5;223")


def _dim(text: str) -> str:
    return _paint(text, "38;5;245")


def _muted(text: str) -> str:
    return _paint(text, "38;5;240")


def _accent(text: str) -> str:
    return _paint(text, "38;5;110")


# ── CLI helpers ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZEROne — Persona Runtime")
    parser.add_argument("--name", default="ZEROne")
    parser.add_argument("--provider", default="deepseek", choices=["openai", "deepseek", "ollama", "codex"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--session", default="main")
    parser.add_argument("--companion-mode", default="companion",
                        choices=["companion", "operator", "teacher", "brainstorm", "reflect"])
    parser.add_argument("--one-shot", default=None, help="Run one prompt and exit")
    parser.add_argument("--workspace", default=None, help="Workspace root directory")
    return parser.parse_args()


def show_banner(z: ZEROne) -> None:
    lines = [
        "",
        _rule("═"),
        f"  {_label(z.name)}  {_dim('Persona Runtime')}",
        _rule(),
        f"  {_accent('provider')}  {z.provider.name}",
        f"  {_accent('mode')}      {z.config.companion_mode}",
        _rule(),
    ]
    print("\n".join(lines))


def show_welcome(z: ZEROne) -> None:
    w = z.welcome_context()
    print(f"  {_paint(w['greeting'], '38;5;223')}")
    print(f"  {_dim('─── suggestions ───')}")
    for s in w["suggestions"]:
        print(f"    {_muted('→')} {_dim(s)}")
    print(f"  {_dim('─── /help ───')}")


def show_help() -> None:
    print(_rule())
    commands = [
        ("/help", "show this"),
        ("/status", "runtime info"),
        ("/history", "recent conversation"),
        ("/provider", "switch provider"),
        ("/model", "switch model"),
        ("/mode", "switch companion mode"),
        ("/session <name>", "switch session"),
        ("/new", "new session"),
        ("/exit", "quit"),
    ]
    for cmd, desc in commands:
        print(f"  {_label(cmd):14} {_dim(desc)}")
    print(_rule())


def show_status(z: ZEROne, session_id: str) -> None:
    s = z.runtime_summary(session_id)
    print(_rule())
    print(f"  {_accent('provider')}  {s['provider']} / {s['model']}")
    print(f"  {_accent('mode')}      {s['mode']}")
    print(f"  {_accent('session')}   {s['session_id']}  ({s['messages']} turns)")
    if s['summary']:
        print(f"  {_accent('summary')}  {s['summary'][:80]}")
    print(_rule())


def show_skills(z: ZEROne) -> None:
    skills = z.list_skills()
    if not skills:
        print(f"  {_dim('No skills loaded.')}")
        return
    print(_rule())
    for s in skills:
        label = "●" if s["active"] else "○"
        tools = ", ".join(s["tools"][:5])
        print(f"  {_muted(label)}  {_accent(s['name'])}  {s['label']}  {_muted('[' + tools + ']')}")
    print(_rule())


def show_memories(z: ZEROne) -> None:
    memories = z.list_memories()
    if not memories:
        print(f"  {_dim('No memories yet.')}")
        return
    print(_rule())
    for m in memories:
        print(f"  {_muted(m['id'][:8])}  {m['text'][:80]}")
    print(_rule())


def show_sessions(z: ZEROne) -> None:
    sessions = z.list_sessions()
    if not sessions:
        print(f"  {_dim('No sessions yet.')}")
        return
    print(_rule())
    for s in sessions:
        print(f"  {_muted('○')}  {s['id']} ({s['count']} turns)")
    print(_rule())


def show_history(z: ZEROne, session_id: str) -> None:
    session = z.load_session(session_id)
    msgs = session.get("messages", [])[-8:]
    print(_rule())
    for m in msgs:
        role = _dim("you →") if m["role"] == "user" else _label(f"{z.name} →")
        content = (m.get("content") or "")[:200]
        print(f"  {role} {content}")
    print(_rule())


def _thinking(stop: threading.Event, status_text: str) -> None:
    frames = ["", ".", "..", "..."]
    i = 0
    while not stop.is_set():
        dots = frames[i % len(frames)]
        sys.stdout.write(f"\r  {_dim(status_text)}{_paint(dots, '38;5;240')}")
        sys.stdout.flush()
        i += 1
        stop.wait(0.35)
    sys.stdout.write("\r" + " " * (len(status_text) + 8) + "\r")
    sys.stdout.flush()


def _typewrite(text: str, delay: float = 0.008) -> None:
    if not text:
        print()
        return
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        if ch not in ("\n", "\r"):
            time.sleep(delay)
    if not text.endswith("\n"):
        print()


def reply_and_show(z: ZEROne, session_id: str, text: str) -> None:
    status_text = f"{z.name} is working" if (z._is_operator_request(text) or z._is_open_request(text)) else f"{z.name} is thinking"
    stop = threading.Event()
    t = threading.Thread(target=_thinking, args=(stop, status_text), daemon=True)
    t.start()
    reply = z.reply(session_id, text, metadata={"surface": "cli"})

    content = str(reply.get("content", ""))
    steps = reply.get("meta", {}).get("agent_steps", [])

    stop.set()
    t.join()

    if steps:
        print(f"  {_accent('steps')}")
        for s in steps:
            tool = s.get("tool", "")
            result = (s.get("result") or "")[:70]
            print(f"    {_dim(tool)}  {_muted(result)}")

    sys.stdout.write(f"  {_label(z.name)}  ")
    sys.stdout.flush()
    _typewrite(content)
    print()


def main() -> int:
    args = parse_args()
    workspace = args.workspace or os.environ.get("MIP_WORKSPACE_ROOT")

    z = ZEROne(Config(
        assistant_name=args.name,
        provider_name=args.provider,
        visual_provider_name=os.environ.get("MIP_VISUAL_PROVIDER", "codex"),
        model=args.model,
        companion_mode=args.companion_mode,
        workspace_root=workspace,
    ))
    session_id = args.session

    if args.one_shot:
        reply_and_show(z, session_id, args.one_shot)
        return 0

    show_banner(z)
    show_welcome(z)
    print()

    while True:
        try:
            raw = input(f"  {_paint('you ›', '38;5;151')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not raw:
            continue
        if raw == "/help":
            show_help(); continue
        if raw == "/status":
            show_status(z, session_id); continue
        if raw == "/history":
            show_history(z, session_id); continue
        if raw == "/memories":
            show_memories(z); continue
        if raw == "/sessions":
            show_sessions(z); continue
        if raw == "/skills":
            show_skills(z); continue
        if raw.startswith("/skill "):
            rest = raw.split(" ", 1)[1].strip()
            if rest.startswith("+"):
                name = rest[1:].strip()
                if z.activate_skill(name):
                    print(f"  {_accent('skill')}  +{name}")
                else:
                    print(f"  {_accent('error')}  skill not found: {name}")
            elif rest.startswith("-"):
                name = rest[1:].strip()
                if z.deactivate_skill(name):
                    print(f"  {_accent('skill')}  -{name}")
                else:
                    print(f"  {_accent('error')}  skill not active: {name}")
            elif rest == "reload":
                z.reload_skills()
                print(f"  {_accent('skill')}  reloaded")
            else:
                z.set_skills([s.strip() for s in rest.split(",")])
                print(f"  {_accent('skill')}  set to {rest}")
            continue
        if raw.startswith("/session "):
            sid = raw.split(" ", 1)[1].strip()
            if sid:
                session_id = sid
                print(f"  {_accent('session')}  {session_id}")
            continue
        if raw.startswith("/new"):
            sid = raw.split(" ", 1)[1].strip() if len(raw.split(" ", 1)) > 1 and raw.split(" ", 1)[1].strip() else "main-fresh"
            session_id = sid
            print(f"  {_accent('session')}  {session_id} (fresh)")
            continue
        if raw == "/provider":
            print(f"  {_accent('provider')}  {z.provider.name} / {getattr(z.provider, 'model', 'default')}")
            continue
        if raw.startswith("/provider "):
            name = raw.split(" ", 1)[1].strip().lower()
            if name not in ("openai", "deepseek", "ollama", "codex"):
                print(f"  {_accent('error')}  provider must be: deepseek, openai, ollama, codex")
                continue
            z.set_provider(name)
            print(f"  {_accent('provider')}  {z.provider.name} / {getattr(z.provider, 'model', 'default')}")
            continue
        if raw.startswith("/model "):
            model = raw.split(" ", 1)[1].strip()
            z.set_model(model) if model else print(f"  {_accent('error')}  model name required")
            continue
        if raw == "/mode":
            print(f"  {_accent('mode')}  {z.config.companion_mode}")
            continue
        if raw.startswith("/mode "):
            mode = raw.split(" ", 1)[1].strip().lower()
            valid = ["companion", "operator", "teacher", "brainstorm", "reflect"]
            if mode in valid:
                z.set_mode(mode)
                print(f"  {_accent('mode')}  {mode}")
            else:
                print(f"  {_accent('error')}  mode must be: {', '.join(valid)}")
            continue
        if raw.startswith("/remember "):
            fact = raw.split(" ", 1)[1].strip()
            if fact:
                r = z.remember(fact)
                print(f"  {_accent('saved')}  {r['id'][:8]}  {fact[:80]}")
            continue
        if raw.startswith("/forget "):
            mid = raw.split(" ", 1)[1].strip()
            if z.forget(mid):
                print(f"  {_accent('forgot')}  {mid}")
            else:
                print(f"  {_accent('error')}  memory not found")
            continue
        if raw == "/exit":
            return 0
        if raw.startswith("/"):
            print(f"  unknown command. /help for available.")
            continue

        reply_and_show(z, session_id, raw)


if __name__ == "__main__":
    raise SystemExit(main())
