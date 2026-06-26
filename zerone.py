"""ZEROne — M.I.P. Runtime."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from providers import BaseProvider, ProviderError, build_provider

from character_adapter import filter_response, load_character_adapter

# ── Utils ──────────────────────────────────────────────────

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
        return lines[-1].strip() if len(lines) == 2 else ""
    return s


def _extract_html(text: str) -> str | None:
    m = re.search(r"```html\s*\n([\s\S]*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    for tag in ("<!DOCTYPE html>", "<html"):
        start = text.find(tag)
        if start >= 0:
            end = text.rfind("</html>")
            if end >= 0:
                return text[start:end + 7].strip()
    return None


# ── Session / Memory stores ────────────────────────────────

class SessionStore:
    """One JSON file per session."""

    def __init__(self, data_dir: Path):
        self.dir = data_dir / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", sid)
        return self.dir / f"{safe}.json"

    def load(self, sid: str) -> dict[str, Any]:
        p = self._path(sid)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                return {"id": sid, "messages": [], "meta": {"corrupt_file": str(p)}}
        return {"id": sid, "messages": [], "meta": {}}

    def save(self, session: dict[str, Any]) -> None:
        self._path(session["id"]).write_text(json.dumps(session, indent=2, default=str))

    def list_sessions(self, limit: int = 20) -> list[dict[str, str | int]]:
        rows: list[dict[str, str | int]] = []
        for p in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            rows.append({"id": p.stem, "count": len(data.get("messages", [])), "updated": str(p.stat().st_mtime)})
            if len(rows) >= limit:
                break
        return rows

    def delete(self, sid: str) -> bool:
        p = self._path(sid)
        if p.exists():
            p.unlink()
            return True
        return False


class MemoryStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                self._data = raw if isinstance(raw, list) else raw.get("memories", [])
            except (json.JSONDecodeError, TypeError):
                self._data = []
        else:
            self._data = []

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2, default=str))

    def add(self, text: str) -> dict[str, Any]:
        record = {"id": str(uuid.uuid4()), "text": text, "created_at": utc_now()}
        self._data.append(record)
        self._save()
        return record

    def list_items(self) -> list[dict[str, Any]]:
        return list(self._data)

    def relevant_items(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            return self._data[:limit]
        words = set(query.lower().split())
        scored: list[tuple[int, dict[str, Any]]] = []
        for m in self._data:
            overlap = len(words & set((m.get("text") or "").lower().split()))
            if overlap:
                scored.append((overlap, m))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored][:limit]

    def delete(self, mid: str) -> bool:
        n = len(self._data)
        self._data = [m for m in self._data if m.get("id") != mid]
        if len(self._data) < n:
            self._save()
            return True
        return False


# ── Tool system ────────────────────────────────────────────

class ToolError(RuntimeError):
    pass


class Registry:
    """Registered callable tools with schema and handler."""

    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}

    def register(self, name: str, schema: dict[str, Any], handler: Callable[..., str]) -> None:
        self._tools[name] = {"schema": schema, "handler": handler}

    def tool(self, name: str, schema: dict[str, Any]):
        """Decorator to register a tool."""
        def wrapper(fn: Callable[..., str]) -> Callable[..., str]:
            self.register(name, schema, fn)
            return fn
        return wrapper

    def get(self, name: str) -> dict[str, Any] | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas_block(self, allowed: set[str] | None = None) -> str:
        lines: list[str] = []
        for name in sorted(self._tools):
            if allowed is not None and name not in allowed:
                continue
            t = self._tools[name]
            s = json.dumps(t["schema"], indent=2)
            lines.append(f'tool "{name}" args: {s}')
        return "\n".join(lines)


# ── Built-in tools ─────────────────────────────────────────

def _build_workspace_tools(root: Path) -> tuple[Registry, dict[str, str]]:
    """Build the tool registry and descriptions for the operator workspace."""
    reg = Registry()
    descs: dict[str, str] = {}

    def _resolve(target: str, wroot: Path = root) -> Path:
        raw = Path(target)
        path = (raw if raw.is_absolute() else wroot / raw).resolve()
        if str(path).startswith(str(wroot)):
            if path.exists():
                return path
            if "/" not in target:
                return (wroot / target).resolve()
        matches = list(wroot.rglob(target))
        if len(matches) == 1:
            return matches[0].resolve()
        if str((wroot / target).resolve()).startswith(str(wroot)):
            return (wroot / target).resolve()
        raise ToolError(f"Cannot resolve path: {target}")

    def _read(path: str, start: int | None = None, end: int | None = None) -> str:
        resolved = _resolve(path)
        text = resolved.read_text()
        if start is not None or end is not None:
            lines = text.splitlines()
            return "\n".join(lines[(start or 1) - 1:end])
        return text

    def _write(path: str, content: str) -> str:
        resolved = _resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return f"wrote {len(content.splitlines())} lines"

    def _replace(path: str, old: str, new: str) -> str:
        resolved = _resolve(path)
        text = resolved.read_text()
        count = text.count(old)
        text = text.replace(old, new, count)
        resolved.write_text(text)
        return f"replaced {count} occurrence(s)"

    def _insert(path: str, anchor: str, content: str, after: bool = True) -> str:
        resolved = _resolve(path)
        text = resolved.read_text()
        if after:
            text = text.replace(anchor, anchor + "\n" + content)
        else:
            text = text.replace(anchor, content + "\n" + anchor)
        resolved.write_text(text)
        return f"inserted {len(content.splitlines())} lines"

    def _list(path: str = ".") -> str:
        resolved = _resolve(path)
        if resolved.is_file():
            return str(resolved)
        files = sorted(str(p.relative_to(root)) for p in resolved.rglob("*"))
        return "\n".join(files[:200])

    def _search(pattern: str, path: str = ".") -> str:
        base = _resolve(path)
        try:
            r = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ToolError(f"Invalid regex: {exc}")
        results: list[str] = []
        targets = sorted(base.rglob("*")) if base.is_dir() else [base]
        for p in targets:
            if not p.is_file():
                continue
            try:
                for i, line in enumerate(p.read_text().splitlines(), 1):
                    if r.search(line):
                        results.append(f"{p.relative_to(root)}:{i}: {line.strip()[:120]}")
            except (OSError, UnicodeDecodeError):
                continue
        return "\n".join(results[:50])

    def _run(cmd: str) -> str:
        import subprocess
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=str(root))
            out = r.stdout + r.stderr
            return out.strip()[:2000] or f"exit code: {r.returncode}"
        except subprocess.TimeoutExpired:
            raise ToolError("Command timed out")

    def _open(target: str) -> str:
        import subprocess
        resolved = _resolve(target)
        uri = str(resolved) if resolved.exists() else target
        subprocess.run(["open", uri], check=True, timeout=5)
        return f"Opened {uri}"

    # Register tools
    reg.register("read_file", {"path": "...", "start?": 1, "end?": 50}, _read)
    reg.register("write_file", {"path": "...", "content": "..."}, _write)
    reg.register("replace_in_file", {"path": "...", "old": "...", "new": "..."}, _replace)
    reg.register("insert_in_file", {"path": "...", "anchor": "...", "content": "...", "after?": True}, _insert)
    reg.register("list_files", {"path?": "."}, _list)
    reg.register("search_text", {"pattern": "...", "path?": "."}, _search)
    reg.register("run_command", {"command": "..."}, _run)
    reg.register("open_target", {"target": "..."}, _open)

    return reg, {}


# ── Persona loading ────────────────────────────────────────

def _load_persona(path: Path) -> dict[str, Any]:
    try:
        spec = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"name": "Assistant", "display_name": "Assistant", "purpose": "Help the user.", "vibe": ["helpful"], "register": "conversational", "formality": 0.5, "emoji_usage": "rare", "raw": {}, "template_path": path.parent / "prompt-template.txt"}
    identity = spec.get("identity", {})
    voice = spec.get("voice", {})

    def g(*keys: str, default: Any = "") -> Any:
        for k in keys:
            val = spec
            for part in k.split("."):
                if isinstance(val, dict):
                    val = val.get(part)
                else:
                    val = None
                    break
            if val:
                return val
        return default

    return {
        "name": identity.get("name", "Assistant"),
        "display_name": identity.get("display_name", identity.get("name", "Assistant")),
        "purpose": identity.get("core_purpose", g("purpose")),
        "essence": identity.get("essence_statement", g("essence")),
        "vibe": identity.get("vibe", ["helpful"]),
        "register": voice.get("register", "conversational"),
        "formality": voice.get("formality", 0.5),
        "emoji_usage": voice.get("emoji_usage", "rare"),
        "raw": spec,
        "template_path": path.parent / "prompt-template.txt",
    }


# ── Skill system (file-based toolkits) ─────────────────────

SKILL_AGENT_PROMPT = """You decide which tool to call next. Return only a JSON object:

- "kind": "tool" or "final"
- "tool": the tool name
- "args": object with the tool's arguments
- "reason": short explanation

Available tools:

{tool_schemas}

When done, return {"kind": "final", "message": "summary"}.

Respond with JSON only. No markdown. No arrays."""

MODE_GUIDANCE: dict[str, str] = {
    "companion": "Stay present and warm. Match the user's pacing. Help them think.",
    "operator": "Be efficient and concrete. Prefer action over discussion.",
    "teacher": "Explain clearly. Encourage understanding, not just answers.",
    "brainstorm": "Be generative. Build on ideas. Stay open.",
    "reflect": "Listen closely. Summarise. Help the user see their own thinking.",
}


# ── Config ─────────────────────────────────────────────────

@dataclass
class Config:
    assistant_name: str = "ZEROne"
    provider_name: str = "deepseek"
    model: str | None = None
    persona_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "persona.zeron.spec.json")
    skills_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "skills")
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    workspace_root: str | None = None
    companion_mode: str = "companion"
    history_window: int = 14


# ── The harness ────────────────────────────────────────────

class ZEROne:
    """Conversation + operator skills, all in one loop."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

        # Persona
        self._persona = _load_persona(self.config.persona_path)
        self.name = self._persona["display_name"]

        # Character adapter (MIP enforcement — persona consistency across models)
        self._adapter = load_character_adapter(str(self.config.persona_path), mode="normal")

        # Provider
        self.provider = build_provider(self.config.provider_name, self.config.model)

        # Workspace
        wroot = Path(self.config.workspace_root or Path.home() / "Documents" / "Playground").resolve()
        self._workspace_root = wroot

        # Tool registry (built-in workspace tools)
        self._tools, _ = _build_workspace_tools(wroot)

        # Skills: loaded from skills/ folder
        self._skills: dict[str, dict[str, Any]] = {}
        self._active_skills: list[str] = []
        self._load_skills()

        # Session memory compounding — carry key context between sessions
        self._session_memory_path = self.config.data_dir / "session-memory.json"
        self._session_memory: dict[str, Any] = self._load_session_memory()

        # Storage
        self._session_store = SessionStore(self.config.data_dir)
        self._memory_store = MemoryStore(self.config.data_dir / "memories.json")

    # ── Session memory compounding ────────────────────────

    def _load_session_memory(self) -> dict[str, Any]:
        try:
            if self._session_memory_path.exists():
                return json.loads(self._session_memory_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
        return {"entries": [], "last_session": None}

    def _save_session_memory(self) -> None:
        self._session_memory_path.write_text(json.dumps(self._session_memory, indent=2, default=str))

    def _summarize_session(self, session: dict[str, Any]) -> str:
        """Create a compact summary of a session for memory carry-over."""
        msgs = session.get("messages", [])
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        if not user_msgs:
            return "(empty session)"
        topics = [m["content"][:100] for m in user_msgs[-3:]]
        return f"Turn count: {len(user_msgs)}. Recent topics: {'; '.join(topics)}"

    def _should_store_session_memory(self, session: dict[str, Any]) -> bool:
        msgs = session.get("messages", [])
        user_count = len([m for m in msgs if m.get("role") == "user"])
        return user_count > 0 and user_count % 5 == 0

    def _store_session_memory_if_due(self, session: dict[str, Any]) -> None:
        """Every 5 turns, save a memory entry."""
        msgs = session.get("messages", [])
        user_count = len([m for m in msgs if m.get("role") == "user"])
        if user_count > 0 and user_count % 5 == 0:
            summary = self._summarize_session(session)
            self._session_memory.setdefault("entries", []).append({
                "session_id": session.get("id", "unknown"),
                "turn": user_count,
                "summary": summary,
                "timestamp": utc_now(),
            })
            self._session_memory["last_session"] = session.get("id", "unknown")
            # Prune to last 20 entries
            if len(self._session_memory["entries"]) > 20:
                self._session_memory["entries"] = self._session_memory["entries"][-20:]
            self._save_session_memory()

    def _session_memory_context(self) -> str:
        """Build a summary of past sessions for context."""
        entries = self._session_memory.get("entries", [])
        if not entries:
            return "(no prior session history)"
        lines = [f"- {e['summary']} ({e['timestamp'][:10]})" for e in entries[-5:]]
        return "Previous sessions:\n" + "\n".join(lines)

    # ── Skills ──────────────────────────────────────────

    def _load_skills(self) -> None:
        """Load all .json skill files from skills/ directory."""
        skills_dir = self.config.skills_dir
        if not skills_dir.exists():
            try:
                skills_dir.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                return  # can't create skills dir, no skills loaded
        for f in sorted(skills_dir.glob("*.json")):
            try:
                spec = json.loads(f.read_text())
                name = spec.get("name", f.stem)
                self._skills[name] = {
                    "name": name,
                    "tools": set(spec.get("tools", [])),
                    "prompt": spec.get("prompt", ""),
                    "label": spec.get("label", name),
                    "file": f,
                }
            except (json.JSONDecodeError, OSError):
                pass

    def reload_skills(self) -> None:
        self._skills.clear()
        self._load_skills()

    def list_skills(self) -> list[dict[str, Any]]:
        return [
            {"name": s["name"], "label": s["label"], "tools": sorted(s["tools"]),
             "active": s["name"] in self._active_skills}
            for s in self._skills.values()
        ]

    def activate_skill(self, name: str) -> bool:
        if name in self._skills and name not in self._active_skills:
            self._active_skills.append(name)
            return True
        return False

    def deactivate_skill(self, name: str) -> bool:
        if name in self._active_skills:
            self._active_skills.remove(name)
            return True
        return False

    def set_skills(self, names: list[str]) -> None:
        self._active_skills = [n for n in names if n in self._skills]

    def _active_tool_names(self) -> set[str]:
        names: set[str] = set()
        for sk in self._active_skills:
            s = self._skills.get(sk)
            if s:
                names.update(s["tools"])
        # Always include core tools
        names.update({"read_file", "write_file", "replace_in_file", "insert_in_file",
                       "list_files", "search_text", "run_command", "open_target"})
        return names

    def _skill_prompts(self) -> str:
        lines: list[str] = []
        for sk in self._active_skills:
            s = self._skills.get(sk)
            if s and s.get("prompt"):
                lines.append(f"[{s['label']}] {s['prompt']}")
        return "\n\n".join(lines)

    # ── Public API ──────────────────────────────────────

    def persona_summary(self) -> dict[str, Any]:
        p = self._persona
        return {
            "name": p["display_name"],
            "purpose": p["purpose"],
            "vibe": p["vibe"],
            "register": p["register"],
        }

    def welcome_context(self) -> dict[str, Any]:
        return {
            "greeting": "I'm here. What shall we work on?",
            "suggestions": (
                "Create a landing page and open it",
                "Improve it dramatically",
                "What's in my workspace?",
            ),
        }

    def runtime_summary(self, session_id: str) -> dict[str, Any]:
        session = self._session_store.load(session_id)
        return {
            "session_id": session_id,
            "messages": len(session.get("messages", [])),
            "provider": self.provider.name,
            "model": getattr(self.provider, "model", "default"),
            "mode": self.config.companion_mode,
            "active_skills": list(self._active_skills),
            "summary": session.get("meta", {}).get("session_summary", ""),
        }

    def set_provider(self, name: str) -> None:
        self.provider = build_provider(name, getattr(self.provider, "model", None))
        self.config.provider_name = name

    def set_model(self, model: str) -> None:
        self.provider.model = model

    def set_mode(self, mode: str) -> None:
        self.config.companion_mode = mode

    def remember(self, text: str) -> dict[str, Any]:
        return self._memory_store.add(text)

    def forget(self, mid: str) -> bool:
        return self._memory_store.delete(mid)

    def list_memories(self) -> list[dict[str, Any]]:
        return self._memory_store.list_items()

    def list_sessions(self, limit: int = 20) -> list[dict[str, str | int]]:
        return self._session_store.list_sessions(limit=limit)

    def load_session(self, session_id: str) -> dict[str, Any]:
        return self._session_store.load(session_id)

    # ── Prompt building ────────────────────────────────

    def _build_prompt(self, user_text: str, session: dict[str, Any]) -> str:
        p = self._persona
        template_path: Path = p.get("template_path", self.config.persona_path.parent / "prompt-template.txt")
        try:
            template = template_path.read_text()
        except FileNotFoundError:
            template = "You are {name}.\n{purpose}\n\nContext:\n{conversation_snapshot}\n\n{memory_context}"

        messages = session.get("messages", [])
        recent = []
        for m in messages[-self.config.history_window:]:
            content = (m.get("content") or "")[:500]
            recent.append(f"{m.get('role', 'user')}: {content}")
        snapshot = "\n".join(recent) if recent else "(no prior conversation)"

        memories = self._memory_store.relevant_items(user_text, limit=5)
        mem_lines = [f"- {mem['text']}" for mem in memories]
        mem_ctx = "\n".join(mem_lines) if mem_lines else "(nothing stored yet)"

        # Session memory compounding
        session_ctx = self._session_memory_context()

        meta = session.get("meta", {})
        summary = meta.get("session_summary", "")

        mode_guide = MODE_GUIDANCE.get(self.config.companion_mode, "I'm here to help.")

        # Skill prompts
        skill_guide = self._skill_prompts()
        tool_names = self._active_tool_names()
        tool_desc = ", ".join(sorted(tool_names)) if tool_names else ""

        try:
            return template.format(
                name=self.name,
                purpose=p.get("purpose", ""),
                essence=p.get("essence", ""),
                register=p.get("register", "conversational"),
                formality=p.get("formality", 0.5),
                emoji_usage=p.get("emoji_usage", "rare"),
                companion_mode=self.config.companion_mode,
                mode_guidance=mode_guide,
                conversation_snapshot=snapshot,
                session_summary=summary or "(in progress)",
                durable_memory=session_ctx,
                relevant_memory=mem_ctx,
                toolkit_guidance=tool_desc,
                skill_guidance=skill_guide,
            )
        except KeyError as exc:
            msg = f"You are {self.name}.\n{p.get('purpose', '')}\n\nContext:\n{snapshot}\n\n{mem_ctx}"
            return msg

    # ── Operator detection ─────────────────────────────

    def _is_operator_request(self, text: str) -> bool:
        lowered = text.lower()
        actions = ("create", "build", "make", "write", "edit", "update", "open", "launch",
                   "fix", "change", "improve", "rewrite", "redesign", "refine", "redo")
        targets = ("file", "page", "landing page", "html", "browser", "folder", "readme",
                   "index", "workspace", "site", "it")
        return any(a in lowered for a in actions) and any(t in lowered for t in targets)

    def _is_open_request(self, text: str) -> bool:
        lowered = text.lower().strip()
        return any(p in lowered for p in ["reopen", "re-open", "open it", "open that",
                                           "show me", "show it", "can you open", "open again",
                                           "open the", "launch it"])

    def _contextualize_request(self, text: str, session: dict[str, Any]) -> str:
        lowered = text.lower().strip()
        last_target = session.get("meta", {}).get("last_operator_target", "")
        if not last_target:
            return text
        if any(p in lowered for p in ["reopen", "re-open", "open it", "open again", "show it", "show me it"]):
            return f"open {last_target}"
        if any(p in lowered for p in ["improve it", "make it better", "second pass", "wow me", "refine it"]):
            return f"improve {last_target}"
        return text

    def _remember_target(self, session: dict[str, Any], steps: list[dict[str, Any]]) -> None:
        for step in steps:
            args = step.get("args", {})
            t = str(args.get("target") or args.get("path") or "").strip()
            if t:
                session.setdefault("meta", {})
                session["meta"]["last_operator_target"] = t
                break

    # ── Agent loop ─────────────────────────────────────

    def _agent_loop(self, task: str, session: dict[str, Any] | None = None, max_steps: int = 6) -> dict[str, Any]:
        history: list[dict[str, str]] = []
        steps: list[dict[str, Any]] = []
        tool_names = self._active_tool_names()

        for _ in range(max_steps):
            messages: list[dict[str, str]] = [{"role": "user", "content": f"Task: {task}"}]

            # Inject recent artifact context into prompt
            if session:
                last_target = session.get("meta", {}).get("last_operator_target", "")
                if last_target:
                    messages[0]["content"] += f"\n\nThe last file you worked on was: {last_target}. Re-open it if needed."

            for h in history:
                messages.append(h)
            schemas = self._tools.schemas_block(tool_names)
            prompt = SKILL_AGENT_PROMPT.replace("{tool_schemas}", schemas)

            try:
                raw = self.provider.generate(prompt, messages)
            except ProviderError as exc:
                return {"message": f"Provider error: {exc}", "steps": steps}

            try:
                plan = json.loads(_strip_fences(raw))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                return {"message": f"Model returned bad JSON: {exc}", "steps": steps}

            if not isinstance(plan, dict):
                return {"message": f"Model returned unexpected format: {str(plan)[:200]}", "steps": steps}

            if "kind" not in plan:
                plan["kind"] = "final" if plan.get("message") else "tool"

            if plan["kind"] == "final":
                return {"message": str(plan.get("message", "")).strip(), "steps": steps}

            tool = str(plan.get("tool", "")).strip()
            if not tool:
                return {"message": "Model did not specify a tool", "steps": steps}
            args = plan.get("args", {}) or {}
            if not isinstance(args, dict):
                args = {}
            result = self._call_tool(tool, args)
            steps.append({"tool": tool, "args": args, "reason": plan.get("reason", ""), "result": result})
            history.append({"role": "assistant", "content": json.dumps({k: v for k, v in plan.items() if k != "kind"})})
            history.append({"role": "user", "content": f"Tool result:\n{result}"})

        return {"message": "Operator reached step limit.", "steps": steps}

    def _call_tool(self, name: str, args: dict[str, Any]) -> str:
        try:
            tool = self._tools.get(name)
            if not tool:
                return f"Unknown tool: {name}"
            return tool["handler"](**args)
        except ToolError as exc:
            return f"Tool error: {exc}"
        except TypeError as exc:
            return f"Arg error: {exc}"

    # ── Main reply ─────────────────────────────────────

    def reply(self, session_id: str, user_text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        session = self._safe_load_session(session_id)
        effective_text = self._contextualize_request(user_text, session)

        # Store memory for facts
        try:
            stored = self._maybe_store_memory(user_text)
        except Exception:
            stored = None

        user_msg: dict[str, Any] = {
            "id": str(uuid.uuid4()), "role": "user", "content": user_text,
            "created_at": utc_now(), "meta": metadata or {},
        }
        if stored:
            user_msg["meta"]["memory_saved"] = {"id": stored["id"], "text": stored["text"]}
        session["messages"].append(user_msg)

        # Generate draft
        model_messages = [{"role": m["role"], "content": m["content"]}
                          for m in session["messages"][-self.config.history_window:]]

        # Context window overflow protection — trim oldest messages if total content is large
        total_chars = sum(len(m.get("content", "")) for m in session["messages"])
        if total_chars > 40000:
            # Keep only the last 8 turns
            overflow = session["messages"][:-8]
            session["messages"] = session["messages"][-8:]
            # Add a note
            session.setdefault("meta", {})["overflow_trimmed"] = len(overflow)
            model_messages = [{"role": m["role"], "content": m["content"]}
                              for m in session["messages"][-self.config.history_window:]]

        try:
            draft = self.provider.generate(self._build_prompt(effective_text, session), model_messages)
        except ProviderError as exc:
            # Save user message but return error gracefully
            assistant_msg: dict[str, Any] = {
                "id": str(uuid.uuid4()), "role": "assistant", "content": f"Sorry — provider error: {exc}",
                "created_at": utc_now(), "meta": {"provider": self.provider.name, "error": str(exc)},
            }
            session["messages"].append(assistant_msg)
            self._update_meta(session, user_text)
            self._session_store.save(session)
            return assistant_msg

        reply_content = draft
        agent_steps: list[dict[str, Any]] = []

        # Run character adapter on the draft
        draft_filtered, draft_report = filter_response(draft, self._adapter, context={"query": user_text})
        consistency_score = round(draft_report.overall_score, 3)
        if not draft_report.overall_passed:
            reply_content = draft_filtered  # use corrected version

        # Agent loop for operator requests
        if self._is_operator_request(effective_text) or self._is_open_request(effective_text):
            loop_result = self._agent_loop(effective_text, session=session)
            if loop_result:
                agent_steps = loop_result.get("steps", [])
                if loop_result.get("message"):
                    reply_content = loop_result["message"]

                # Fallback: if model generated HTML content, write it
                if not agent_steps and any(w in effective_text.lower() for w in ["create", "build", "make"]):
                    html = _extract_html(draft)
                    if html:
                        try:
                            self._call_tool("write_file", {"path": "index.html", "content": html})
                            self._call_tool("open_target", {"target": "index.html"})
                            agent_steps = [{"tool": "write_file", "args": {"path": "index.html"}, "reason": "write generated HTML", "result": f"wrote {len(html)} chars"}]
                            reply_content = "Created and opened index.html in the browser."
                        except ToolError as exc:
                            agent_steps = [{"tool": "write_file", "args": {"path": "index.html"}, "result": str(exc)}]

                # Run character adapter on agent loop result too
                if reply_content != "Operator reached step limit.":
                    loop_filtered, loop_report = filter_response(reply_content, self._adapter, context={"query": user_text})
                    if loop_report.overall_passed:
                        reply_content = loop_filtered
                    consistency_score = round(loop_report.overall_score, 3)

        meta: dict[str, Any] = {
            "provider": self.provider.name,
            "mode": self.config.companion_mode,
            "active_skills": list(self._active_skills),
            "consistency": consistency_score,
        }
        if agent_steps:
            meta["agent_steps"] = agent_steps

        assistant_msg: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": reply_content,
            "created_at": utc_now(),
            "meta": meta,
        }
        session["messages"].append(assistant_msg)

        self._remember_target(session, agent_steps)
        self._update_meta(session, user_text)
        self._store_session_memory_if_due(session)
        self._session_store.save(session)

        return assistant_msg

    def _safe_load_session(self, session_id: str) -> dict[str, Any]:
        try:
            return self._session_store.load(session_id)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            return {"id": session_id, "messages": [], "meta": {"load_error": str(exc)}}

    def _maybe_store_memory(self, text: str) -> dict[str, Any] | None:
        lowered = text.lower().strip()
        if "?" in text:
            return None
        for pat in [r"^(my name is .+)", r"^(call me .+)", r"^(i (?:like|work on|am working on|prefer) .+)",
                     r"^(my project is .+)"]:
            m = re.match(pat, lowered)
            if m:
                cleaned = " ".join(text.split())[:180]
                return self._memory_store.add(cleaned)
        return None

    def _update_meta(self, session: dict[str, Any], user_text: str) -> None:
        session.setdefault("meta", {})
        msgs = session.get("messages", [])
        session["meta"]["turn_count"] = len([m for m in msgs if m.get("role") == "user"])
        session["meta"]["last_topic"] = user_text[:120]
        user_msgs = [m["content"][:100] for m in msgs if m.get("role") == "user"][-3:]
        if user_msgs:
            session["meta"]["session_summary"] = "Recent topics: " + "; ".join(user_msgs)
            session["meta"]["message_count"] = len(msgs)
