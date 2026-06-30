"""ZEROne — M.I.P. Runtime."""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib import error, parse, request as urllib_request

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


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _extract_dsml_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    invoke_pattern = re.compile(
        r"<｜｜DSML｜｜invoke name=\"([^\"]+)\">(.*?)</｜｜DSML｜｜invoke>",
        re.DOTALL,
    )
    param_pattern = re.compile(
        r"<｜｜DSML｜｜parameter name=\"([^\"]+)\"(?:\s+string=\"true\")?>(.*?)</｜｜DSML｜｜parameter>",
        re.DOTALL,
    )
    for match in invoke_pattern.finditer(text):
        name = match.group(1).strip()
        block = match.group(2)
        args: dict[str, Any] = {}
        for p_match in param_pattern.finditer(block):
            key = p_match.group(1).strip()
            value = p_match.group(2).strip()
            args[key] = value
        if name:
            calls.append({"tool": name, "args": args})
    return calls


def _tool_result_failed(result: str) -> bool:
    lowered = (result or "").lower()
    return lowered.startswith("arg error:") or lowered.startswith("tool error:") or lowered.startswith("unknown tool:")


def _looks_like_code_dump(text: str) -> bool:
    if not text:
        return False
    if _extract_html(text) is not None:
        return True
    return bool(re.search(r"```(?:html|css|js|javascript|json|python)?", text, re.IGNORECASE))


def _clean_visual_query(text: str) -> str:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", text.lower())
    stopwords = {
        "create", "build", "make", "write", "improve", "redesign", "refine", "rewrite",
        "landing", "page", "site", "browser", "html", "with", "from", "into", "that",
        "this", "your", "their", "there", "please", "need", "want", "good", "better",
        "premium", "design", "hero", "section", "background", "brand", "style", "look",
        "feel", "using", "about", "calm", "nice",
    }
    kept: list[str] = []
    for word in words:
        if word in stopwords or len(word) < 3:
            continue
        kept.append(word)
        if len(kept) >= 6:
            break
    return " ".join(kept)


def _strip_html_tags(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = cleaned.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_url(text: str) -> str | None:
    match = re.search(r"https?://[^\s)>\"]+", text)
    return match.group(0) if match else None


def _collect_unique_matches(pattern: str, text: str, limit: int = 12) -> list[str]:
    values: list[str] = []
    for raw in re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL):
        cleaned = _strip_html_tags(raw) if "<" in raw else raw.strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–|:;,")
        if cleaned and cleaned not in values:
            values.append(cleaned)
        if len(values) >= limit:
            break
    return values


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
        cleaned = str(target).strip()
        try:
            resolved = _resolve(cleaned)
            uri = str(resolved)
            subprocess.run(["open", uri], check=True, timeout=5)
            return f"Opened {uri}"
        except ToolError:
            looks_like_app = cleaned.endswith(".app") or ("/" not in cleaned and "." not in cleaned)
            if looks_like_app:
                subprocess.run(["open", "-a", cleaned], check=True, timeout=5)
                return f"Opened application {cleaned}"
            raw = Path(cleaned).expanduser()
            if raw.exists():
                uri = str(raw.resolve())
                subprocess.run(["open", uri], check=True, timeout=5)
                return f"Opened {uri}"
            subprocess.run(["open", cleaned], check=True, timeout=5)
            return f"Opened {cleaned}"

    def _web_search(query: str, max_results: int = 5) -> str:
        cleaned_query = str(query).strip()
        if not cleaned_query:
            raise ToolError("Query is required")

        limit = max(1, min(int(max_results), 8))
        url = "https://html.duckduckgo.com/html/?" + parse.urlencode({"q": cleaned_query})
        req = urllib_request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ZEROne/1.0; +https://github.com/tgoapple/zerone)",
            },
            method="GET",
        )
        try:
            with urllib_request.urlopen(req, timeout=20) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            raise ToolError(f"Web search failed: {exc}") from exc

        matches = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        results: list[str] = []
        for href, title_html in matches:
            title = _strip_html_tags(title_html)
            if not title:
                continue
            results.append(f"{title}\n{href}")
            if len(results) >= limit:
                break

        if not results:
            raise ToolError("No web results found")
        return "\n\n".join(results)

    def _fetch_url(url: str, max_chars: int = 8000) -> str:
        cleaned_url = str(url).strip()
        if not cleaned_url.startswith(("http://", "https://")):
            raise ToolError("URL must start with http:// or https://")

        limit = max(1000, min(int(max_chars), 20000))
        req = urllib_request.Request(
            cleaned_url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ZEROne/1.0; +https://github.com/tgoapple/zerone)",
            },
            method="GET",
        )
        try:
            with urllib_request.urlopen(req, timeout=20) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            raise ToolError(f"URL fetch failed: {exc}") from exc

        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        desc_match = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        headings = re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", html, flags=re.IGNORECASE | re.DOTALL)
        classes = re.findall(r'class=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)

        class_tokens: list[str] = []
        for class_group in classes:
            for token in class_group.split():
                token = token.strip()
                if token and token not in class_tokens:
                    class_tokens.append(token)
                if len(class_tokens) >= 40:
                    break
            if len(class_tokens) >= 40:
                break

        summary_parts = [
            f"URL: {cleaned_url}",
            f"Title: {_strip_html_tags(title_match.group(1)) if title_match else '(none)'}",
            f"Description: {desc_match.group(1).strip() if desc_match else '(none)'}",
        ]
        if headings:
            summary_parts.append("Headings:\n" + "\n".join(f"- {_strip_html_tags(h)[:160]}" for h in headings[:12]))
        if class_tokens:
            summary_parts.append("Class tokens:\n" + ", ".join(class_tokens))
        snippet = _strip_html_tags(html)[:limit]
        if snippet:
            summary_parts.append("Text snippet:\n" + snippet)
        return "\n\n".join(summary_parts)

    # Register tools
    reg.register("read_file", {"path": "...", "start?": 1, "end?": 50}, _read)
    reg.register("write_file", {"path": "...", "content": "..."}, _write)
    reg.register("replace_in_file", {"path": "...", "old": "...", "new": "..."}, _replace)
    reg.register("insert_in_file", {"path": "...", "anchor": "...", "content": "...", "after?": True}, _insert)
    reg.register("list_files", {"path?": "."}, _list)
    reg.register("search_text", {"pattern": "...", "path?": "."}, _search)
    reg.register("web_search", {"query": "...", "max_results?": 5}, _web_search)
    reg.register("fetch_url", {"url": "...", "max_chars?": 8000}, _fetch_url)
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

SKILL_AGENT_PROMPT = """You complete tasks by using tools. You must call at least one tool before you can say you're done. Do not describe what you will do — do it.

Return only a JSON object:

- "kind": "tool" or "final"
- "tool": the tool name (must be one of the available tools)
- "args": object with the tool's arguments
- "reason": short explanation of what this step does

Available tools:

{tool_schemas}

Rules:
- Never return "final" without having called at least one tool.
- Never describe a plan. Execute it.
- If the task is to create or improve a landing page, prefer design_landing_page.
- If the task involves creating or editing a file, use write_file or replace_in_file.
- If the task says "open", use open_target after writing.
- Never emit DSML, XML-like tool markup, or pseudo function-calling syntax.

Respond with JSON only. No markdown. No prose. No arrays."""

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
    visual_provider_name: str = "codex"
    model: str | None = None
    persona_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "persona.zeron.spec.json")
    skills_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "skills")
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    workspace_root: str | None = None
    companion_mode: str = "companion"
    history_window: int = 14


@dataclass(frozen=True)
class ToolkitRoute:
    skill_name: str
    signals: tuple[str, ...]
    actions: tuple[str, ...] = ()
    followup_suffixes: tuple[str, ...] = ()
    followup_actions: tuple[str, ...] = ()
    score: int = 10


TOOLKIT_ROUTES: tuple[ToolkitRoute, ...] = (
    ToolkitRoute(
        skill_name="reference-browser",
        signals=("http://", "https://", "reference site", "reference url", "like this site", "inspired by this site"),
        actions=("inspect", "analyze", "study", "rebuild", "redesign", "match", "influence"),
        score=18,
    ),
    ToolkitRoute(
        skill_name="research",
        signals=("search the web", "search web", "look up", "lookup", "find online", "research", "latest", "news", "current"),
        actions=("search", "look", "find", "research", "check", "compare"),
        score=14,
    ),
    ToolkitRoute(
        skill_name="design",
        signals=("landing page", "homepage", "hero section", "visual design", "ui design", "premium", "brand"),
        actions=("create", "build", "make", "improve", "redesign", "refine", "rewrite"),
        followup_suffixes=(".html",),
        followup_actions=("improve", "refine", "redesign", "another pass", "better", "premium"),
        score=15,
    ),
    ToolkitRoute(
        skill_name="landing-pages",
        signals=("landing page", "homepage", "hero section", "browser", "html", "site"),
        actions=("create", "build", "make", "improve", "refine", "redesign", "rewrite", "wow me"),
        followup_suffixes=(".html",),
        followup_actions=("open", "show", "improve", "refine", "redesign", "rewrite", "second pass", "wow me"),
        score=30,
    ),
    ToolkitRoute(
        skill_name="debug",
        signals=("debug", "bug", "error", "failing", "traceback", "stack trace", "fix this"),
        actions=("fix", "debug", "investigate", "diagnose"),
        followup_suffixes=(".py", ".js", ".ts", ".tsx", ".jsx"),
        followup_actions=("fix", "debug", "why", "investigate"),
        score=22,
    ),
    ToolkitRoute(
        skill_name="web-dev",
        signals=("css", "javascript", "frontend", "component", "layout", "responsive", "ui"),
        actions=("build", "style", "design", "improve", "adjust", "refine"),
        followup_suffixes=(".html", ".css", ".js"),
        followup_actions=("style", "adjust", "improve", "refine"),
        score=16,
    ),
)

VISUAL_REQUEST_SIGNALS: tuple[str, ...] = (
    "image",
    "logo",
    "icon",
    "svg",
    "poster",
    "illustration",
    "render",
    "cover art",
    "album art",
    "brand mark",
    "wordmark",
    "thumbnail",
    "hero image",
    "mockup",
    "visual",
)


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
        self._provider_models: dict[str, str | None] = {
            self.provider.name: getattr(self.provider, "model", None),
        }

        # Workspace
        wroot = Path(self.config.workspace_root or Path.home() / "Documents" / "Playground").resolve()
        self._workspace_root = wroot

        # Tool registry (built-in workspace tools)
        self._tools, _ = _build_workspace_tools(wroot)
        self._register_creative_tools()

        # Skills: loaded from skills/ folder
        self._skills: dict[str, dict[str, Any]] = {}
        self._active_skills: list[str] = []
        self._load_skills()
        # Activate default skills
        for dskill in ["session-memory", "llm-wiki", "pi-design", "web-dev", "pi-core"]:
            self.activate_skill(dskill)

        # Session memory compounding — carry key context between sessions
        self._session_memory_path = self.config.data_dir / "session-memory.json"
        self._session_memory: dict[str, Any] = self._load_session_memory()

        # Storage
        self._session_store = SessionStore(self.config.data_dir)
        self._memory_store = MemoryStore(self.config.data_dir / "memories.json")

    def _register_creative_tools(self) -> None:
        def _inspect_reference_url(url: str, max_chars: int = 8000) -> str:
            cleaned_url = str(url).strip()
            if not cleaned_url.startswith(("http://", "https://")):
                return self._call_tool("fetch_url", {"url": cleaned_url, "max_chars": max_chars})

            script_path = Path(__file__).resolve().parent / "reference_inspector.js"
            try:
                result = subprocess.run(
                    ["node", str(script_path), cleaned_url],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(self._workspace_root),
                )
            except subprocess.TimeoutExpired as exc:
                return f"Tool error: Reference inspection timed out: {exc}"

            if result.returncode != 0 or not result.stdout.strip():
                fallback = self._call_tool("fetch_url", {"url": cleaned_url, "max_chars": max_chars})
                if not _tool_result_failed(fallback):
                    return fallback
                stderr = (result.stderr or "").strip()
                return f"Tool error: Reference inspection failed. {stderr[:400]}"

            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                fallback = self._call_tool("fetch_url", {"url": cleaned_url, "max_chars": max_chars})
                if not _tool_result_failed(fallback):
                    return fallback
                return f"Tool error: Reference inspection returned invalid JSON: {result.stdout[:240]}"

            hero = payload.get("hero") or {}
            parts = [
                f"URL: {cleaned_url}",
                f"Title: {str(payload.get('title') or '(none)').strip()}",
                f"Description: {str(payload.get('description') or '(none)').strip()}",
                f"Body background: {str(payload.get('bodyBackground') or '(unknown)').strip()}",
                f"Body text color: {str(payload.get('bodyColor') or '(unknown)').strip()}",
                f"Body font: {str(payload.get('bodyFont') or '(unknown)').strip()}",
                f"Dark page: {'yes' if payload.get('bodyIsDark') else 'no'}",
                f"Section count: {int(payload.get('sectionCount') or 0)}",
                f"Image count: {int(payload.get('imageCount') or 0)}",
            ]
            if hero:
                hero_lines = [
                    f"- selector: {hero.get('selector', '(unknown)')}",
                    f"- height: {hero.get('height', '(unknown)')}px",
                    f"- heading: {hero.get('heading', '(none)')}",
                    f"- background color: {hero.get('backgroundColor', '(unknown)')}",
                    f"- background image: {hero.get('backgroundImage', '(none)')}",
                    f"- text align: {hero.get('textAlign', '(unknown)')}",
                    f"- classes: {hero.get('className', '(none)')}",
                ]
                parts.append("Hero cues:\n" + "\n".join(hero_lines))
            headings = payload.get("headings") or []
            if headings:
                parts.append("Headings:\n" + "\n".join(f"- {str(h)[:160]}" for h in headings[:12]))
            class_tokens = payload.get("classTokens") or []
            if class_tokens:
                parts.append("Class tokens:\n" + ", ".join(str(token) for token in class_tokens[:40]))
            links = payload.get("links") or []
            if links:
                parts.append("Link labels:\n" + "\n".join(f"- {str(link)[:120]}" for link in links[:12]))
            snippet = str(payload.get("snippet") or "").strip()
            if snippet:
                parts.append("Text snippet:\n" + snippet[:max(1000, min(int(max_chars), 12000))])
            return "\n\n".join(parts)

        def _design_landing_page(
            path: str,
            brief: str,
            mode: str = "create",
            open_after: bool = True,
            style: str = "auto",
        ) -> str:
            current_html = None
            if mode == "improve":
                try:
                    current_html = self._call_tool("read_file", {"path": path})
                except Exception:
                    current_html = None
            html = self._generate_page_html(brief, path, current_html=current_html, style=style, specialist=True)
            write_result = self._call_tool("write_file", {"path": path, "content": html})
            opened = ""
            if open_after:
                opened = self._call_tool("open_target", {"target": path})
            status = f"{mode}d landing page at {path}; {write_result}"
            if opened:
                status += f"; {opened}"
            return status

        def _build_landing_page(
            path: str,
            brief: str,
            mode: str = "create",
            open_after: bool = True,
        ) -> str:
            return _design_landing_page(path, brief, mode=mode, open_after=open_after, style="auto")

        self._tools.register(
            "design_landing_page",
            {
                "path": "...",
                "brief": "...",
                "mode?": "create|improve",
                "style?": "auto|editorial|product|luxury|minimal|bold",
                "open_after?": True,
            },
            _design_landing_page,
        )

        self._tools.register(
            "inspect_reference_url",
            {
                "url": "...",
                "max_chars?": 8000,
            },
            _inspect_reference_url,
        )

        self._tools.register(
            "build_landing_page",
            {
                "path": "...",
                "brief": "...",
                "mode?": "create|improve",
                "open_after?": True,
            },
            _build_landing_page,
        )

        # ── Knowledge base (wiki) tools ────────────────
        def _wiki_add(title: str, content: str, tags: str = "") -> str:
            wiki_dir = self._workspace_root / "llm-wiki"
            wiki_dir.mkdir(parents=True, exist_ok=True)
            slug = title.lower().replace(" ", "-").replace("/", "-")[:60]
            path = wiki_dir / f"{slug}.md"
            tag_line = f"tags: {tags}\n" if tags else ""
            entry = f"# {title}\n\n{tag_line}{content}\n"
            path.write_text(entry)
            return f"Saved wiki entry '{title}' to {path}"

        def _wiki_search(query: str) -> str:
            wiki_dir = self._workspace_root / "llm-wiki"
            if not wiki_dir.exists():
                return "No wiki entries yet."
            results = []
            for f in sorted(wiki_dir.glob("*.md")):
                text = f.read_text()
                if query.lower() in text.lower():
                    title = text.split("\n")[0].lstrip("# ").strip()
                    results.append(f"- {title} ({f.name})")
            if not results:
                return f"No wiki entries matching '{query}'."
            return "Wiki entries:\n" + "\n".join(results[-10:])

        def _wiki_list() -> str:
            wiki_dir = self._workspace_root / "llm-wiki"
            if not wiki_dir.exists():
                return "No wiki entries yet."
            entries = []
            for f in sorted(wiki_dir.glob("*.md")):
                title = f.read_text().split("\n")[0].lstrip("# ").strip()
                entries.append(f"- {title} ({f.name})")
            if not entries:
                return "No wiki entries yet."
            return "Wiki entries:\n" + "\n".join(entries)

        self._tools.register(
            "wiki_add",
            {"title": "...", "content": "...", "tags?": "..."},
            _wiki_add,
        )
        self._tools.register(
            "wiki_search",
            {"query": "..."},
            _wiki_search,
        )
        self._tools.register(
            "wiki_list",
            {},
            _wiki_list,
        )

        # ── Session memory tools ────────────────────────
        def _recall_session() -> str:
            ctx = self._session_memory_context()
            return ctx

        def _save_memory(text: str) -> str:
            result = self.remember(text)
            return f"Saved: {result.get('id', 'unknown')}"

        def _list_memories() -> str:
            mems = self.list_memories()
            if not mems:
                return "No memories stored."
            return "\n".join(f"- [{m.get('id','?')[:8]}] {m.get('content','')[:120]}" for m in mems[-10:])

        self._tools.register(
            "recall_session",
            {},
            _recall_session,
        )
        self._tools.register(
            "save_memory",
            {"text": "..."},
            _save_memory,
        )
        self._tools.register(
            "list_memories",
            {},
            _list_memories,
        )

        # ── Design polish tools ─────────────────────────
        def _polish_page(path: str) -> str:
            content = self._call_tool("read_file", {"path": path})
            if _tool_result_failed(content):
                return content
            if "<html" not in content and "<!DOCTYPE" not in content:
                return "Not an HTML file."
            prompt = (
                f"Polish this HTML page to feel editorial and intentional. "
                f"Use a dark palette (graphite/ultraviolet/cobalt). "
                f"Improve typography, spacing, and visual hierarchy. "
                f"Keep all existing content. Return only the full HTML.\n\n{content[:6000]}"
            )
            polished = self._generate_page_html(
                prompt, path, current_html=content, style="editorial", specialist=False
            )
            self._call_tool("write_file", {"path": path, "content": polished})
            self._call_tool("open_target", {"target": path})
            return f"Polished {path} with editorial styling and opened in browser."

        self._tools.register(
            "polish_page",
            {"path": "..."},
            _polish_page,
        )

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

    def _tool_names_for_skills(self, skill_names: list[str] | None = None) -> set[str]:
        names: set[str] = set()
        for sk in skill_names if skill_names is not None else self._active_skills:
            s = self._skills.get(sk)
            if s:
                names.update(s["tools"])
        # Always include core tools
        names.update({"read_file", "write_file", "replace_in_file", "open_target"})
        return names

    def _active_tool_names(self) -> set[str]:
        return self._tool_names_for_skills()

    def _skill_prompts(self, skill_names: list[str] | None = None) -> str:
        lines: list[str] = []
        for sk in skill_names if skill_names is not None else self._active_skills:
            s = self._skills.get(sk)
            if s and s.get("prompt"):
                lines.append(f"[{s['label']}] {s['prompt']}")
        return "\n\n".join(lines)

    def _toolkit_route_score(
        self,
        route: ToolkitRoute,
        user_text: str,
        session: dict[str, Any] | None = None,
    ) -> int:
        if route.skill_name not in self._skills:
            return 0

        lowered = user_text.lower()
        last_target = ""
        if session:
            last_target = str(session.get("meta", {}).get("last_operator_target", "")).lower()

        score = 0
        if route.signals and any(signal in lowered for signal in route.signals):
            score += route.score
            if route.actions and any(action in lowered for action in route.actions):
                score += 6

        if (
            last_target
            and route.followup_suffixes
            and any(last_target.endswith(suffix) for suffix in route.followup_suffixes)
            and route.followup_actions
            and any(action in lowered for action in route.followup_actions)
        ):
            score += route.score + 8

        return score

    def _rank_toolkit_routes(
        self,
        user_text: str,
        session: dict[str, Any] | None = None,
    ) -> list[tuple[str, int]]:
        ranked: list[tuple[str, int]] = []
        for route in TOOLKIT_ROUTES:
            score = self._toolkit_route_score(route, user_text, session=session)
            if score > 0:
                ranked.append((route.skill_name, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def _auto_skill_names(self, user_text: str, session: dict[str, Any] | None = None) -> list[str]:
        names: list[str] = []
        for name, _score in self._rank_toolkit_routes(user_text, session=session):
            if name not in self._active_skills and name not in names:
                names.append(name)
            if len(names) >= 2:
                break
        return names

    def _effective_skill_names(self, user_text: str, session: dict[str, Any] | None = None) -> list[str]:
        effective = list(self._active_skills)
        for name in self._auto_skill_names(user_text, session=session):
            if name not in effective:
                effective.append(name)
        return effective

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
            "greeting": "I'm here. What are we building?",
            "suggestions": (
                "create a landing page and open it",
                "improve it — make it better",
                "what's in my workspace?",
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
        self._provider_models[self.provider.name] = getattr(self.provider, "model", None)
        self.provider = build_provider(name, self._provider_models.get(name))
        self._provider_models[self.provider.name] = getattr(self.provider, "model", None)
        self.config.provider_name = name

    def set_model(self, model: str) -> None:
        self.provider.model = model
        self._provider_models[self.provider.name] = model

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

    def _build_prompt(self, user_text: str, session: dict[str, Any], skill_names: list[str] | None = None) -> str:
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
        skill_guide = self._skill_prompts(skill_names)
        tool_names = self._tool_names_for_skills(skill_names)
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

    def _is_visual_request(self, user_text: str) -> bool:
        lowered = user_text.lower()
        if not _contains_any(lowered, VISUAL_REQUEST_SIGNALS):
            return False
        if _contains_any(lowered, ("landing page", "homepage", "page", "site", "html", "browser")):
            return False
        return _contains_any(lowered, ("create", "make", "design", "generate", "draw", "craft", "build", "need", "want"))

    def _provider_for_request(self, user_text: str) -> BaseProvider:
        if not self._is_visual_request(user_text):
            return self.provider
        preferred = (self.config.visual_provider_name or "codex").strip().lower()
        if preferred == self.provider.name:
            return self.provider
        try:
            routed = build_provider(preferred, self._provider_models.get(preferred))
        except ProviderError:
            return self.provider
        self._provider_models[routed.name] = getattr(routed, "model", None)
        return routed

    # ── Operator detection ─────────────────────────────

    def _is_operator_request(self, text: str) -> bool:
        lowered = text.lower()
        actions = ("create", "build", "make", "write", "edit", "update", "open", "launch",
                   "fix", "change", "improve", "rewrite", "redesign", "refine", "redo", "pass")
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
        if "show me" in lowered and any(p in lowered for p in ["workspace", "folder", "files", "what's in"]):
            return text
        open_phrases = (
            "reopen", "re-open", "open it", "open again", "show it", "show me it",
            "show me that", "show me the page", "show me the site", "show me the file",
            "show me the second phase", "show me the improved version", "open the page",
            "open the site", "open the file", "open the second phase", "open the improved version",
            "show me in a browser", "open it in my browser", "open it in the browser",
        )
        improve_phrases = (
            "improve it", "make it better", "second pass", "wow me", "refine it",
            "improve that", "improve the page", "improve the site", "redo it",
            "rewrite it", "redesign it", "take another pass", "another pass", "one more pass",
            "pass again", "make it feel more premium",
        )
        if any(p in lowered for p in open_phrases):
            return f"open {last_target}"
        if any(p in lowered for p in improve_phrases):
            return f"improve {last_target} and open it"
        return text

    def _remember_target(self, session: dict[str, Any], steps: list[dict[str, Any]]) -> None:
        for step in steps:
            args = step.get("args", {})
            t = str(args.get("target") or args.get("path") or "").strip()
            if t:
                session.setdefault("meta", {})
                session["meta"]["last_operator_target"] = t
                break

    def _default_page_target(self) -> str:
        preferred = self._workspace_root / "mip-framework"
        if preferred.exists():
            return "mip-framework/index.html"
        return "index.html"

    def _operator_target(self, task: str, session: dict[str, Any] | None = None) -> str:
        quoted = re.findall(r"[\w./-]+\.(?:html?|css|js|md|txt)", task, flags=re.IGNORECASE)
        if quoted:
            return quoted[0]
        if session:
            last_target = str(session.get("meta", {}).get("last_operator_target", "")).strip()
            if last_target:
                return last_target
        if _contains_any(task, ("landing page", "page", "site", "browser", "html")):
            return self._default_page_target()
        return ""

    def _fallback_landing_page_html(self, name: str) -> str:
        title = name or self.name
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --ink: #1b1a17;
      --muted: #5b554b;
      --panel: rgba(255, 252, 247, 0.78);
      --line: rgba(27, 26, 23, 0.12);
      --accent: #c46a3a;
      --accent-2: #2f6c64;
      --shadow: 0 24px 80px rgba(27, 26, 23, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(196, 106, 58, 0.18), transparent 28%),
        radial-gradient(circle at bottom right, rgba(47, 108, 100, 0.18), transparent 26%),
        linear-gradient(180deg, #fbf6ee 0%, var(--bg) 100%);
      min-height: 100vh;
    }}
    .shell {{
      width: min(1120px, calc(100% - 40px));
      margin: 32px auto;
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background: var(--panel);
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow);
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 12px;
      color: var(--muted);
    }}
    .hero {{
      padding: 72px 0 56px;
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 32px;
      align-items: end;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(3rem, 8vw, 6.4rem);
      line-height: 0.95;
      letter-spacing: -0.05em;
      max-width: 8ch;
    }}
    .lead {{
      max-width: 34rem;
      font-size: 1.15rem;
      line-height: 1.7;
      color: var(--muted);
      margin: 20px 0 0;
    }}
    .card {{
      padding: 22px;
      border-radius: 24px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(27, 26, 23, 0.08);
    }}
    .pulse {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: 0 0 0 12px rgba(196, 106, 58, 0.12);
      margin-bottom: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 18px;
    }}
    .grid h3 {{
      margin: 0 0 8px;
      font-size: 1.05rem;
    }}
    .grid p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }}
    .actions {{
      display: flex;
      gap: 14px;
      margin-top: 28px;
      flex-wrap: wrap;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 48px;
      padding: 0 18px;
      border-radius: 999px;
      text-decoration: none;
      border: 1px solid transparent;
      color: #fff8f2;
      background: var(--ink);
    }}
    .button.alt {{
      color: var(--ink);
      background: transparent;
      border-color: var(--line);
    }}
    @media (max-width: 820px) {{
      .hero, .grid {{
        grid-template-columns: 1fr;
      }}
      .shell {{
        width: min(100% - 24px, 1120px);
        padding: 18px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <div class="topbar">
      <span>{title}</span>
      <span>Companion + Operator</span>
    </div>
    <section class="hero">
      <div>
        <h1>{title} builds with you.</h1>
        <p class="lead">A companion that can think, write, edit, and act. Calm enough to stay with the work. Capable enough to move it forward.</p>
        <div class="actions">
          <a class="button" href="#details">See the flow</a>
          <a class="button alt" href="#capabilities">Explore capabilities</a>
        </div>
      </div>
      <aside class="card">
        <div class="pulse"></div>
        <strong>Designed to feel present.</strong>
        <p class="lead">The interface stays simple while the system underneath handles memory, tools, and real execution.</p>
      </aside>
    </section>
    <section id="capabilities" class="grid">
      <article class="card">
        <h3>Companion voice</h3>
        <p>Warm, steady, and direct over long conversations without falling into template-sounding replies.</p>
      </article>
      <article class="card">
        <h3>Operator actions</h3>
        <p>Creates files, improves drafts, and opens the result instead of talking around the work.</p>
      </article>
      <article class="card">
        <h3>Real continuity</h3>
        <p>Remembers the last artifact so follow-up requests like “show me it” or “second pass” stay grounded.</p>
      </article>
    </section>
  </main>
</body>
</html>"""

    def _derive_unsplash_query(self, task: str, target: str) -> str:
        query = _clean_visual_query(task)
        if query:
            return query
        target_hint = _clean_visual_query(Path(target).stem.replace("-", " ").replace("_", " "))
        return target_hint or "editorial lifestyle"

    def _design_direction(self, task: str, target: str, style: str = "auto") -> dict[str, str]:
        lowered = f"{task} {target}".lower()
        forced = (style or "auto").strip().lower()

        presets: dict[str, dict[str, str]] = {
            "editorial": {
                "name": "editorial",
                "guidance": "Use a restrained editorial feel: elegant typography, asymmetric composition, confident whitespace, quiet luxury, and strong image-led storytelling.",
            },
            "product": {
                "name": "product",
                "guidance": "Use a premium product-marketing feel: crisp hierarchy, sharp demos, clean conversion moments, and restrained supporting sections.",
            },
            "luxury": {
                "name": "luxury",
                "guidance": "Use a luxury brand feel: refined restraint, rich materials, calmer pacing, elevated serif/sans pairing, and understated polish over noise.",
            },
            "minimal": {
                "name": "minimal",
                "guidance": "Use a severe minimal direction: fewer sections, harder editing, cleaner typography, deliberate spacing, and no decorative clutter.",
            },
            "bold": {
                "name": "bold",
                "guidance": "Use a bold contemporary direction: higher contrast, stronger type, distinctive layout shifts, and a more assertive visual rhythm.",
            },
        }

        if forced in presets:
            return presets[forced]
        if any(word in lowered for word in ("luxury", "premium", "elegant", "high-end", "quiet luxury", "fashion", "hotel", "fragrance", "skincare")):
            return presets["luxury"]
        if any(word in lowered for word in ("minimal", "clean", "simple", "quiet", "calm", "monastic")):
            return presets["minimal"]
        if any(word in lowered for word in ("bold", "energetic", "experimental", "campaign", "poster")):
            return presets["bold"]
        if any(word in lowered for word in ("app", "saas", "dashboard", "product", "platform", "tool")):
            return presets["product"]
        return presets["editorial"]

    def _fetch_unsplash_reference(self, task: str, target: str) -> dict[str, str] | None:
        access_key = os.environ.get("UNSPLASH_ACCESS_KEY") or os.environ.get("MIP_UNSPLASH_ACCESS_KEY") or ""
        if not access_key:
            return None

        query = self._derive_unsplash_query(task, target)
        url = "https://api.unsplash.com/search/photos?" + parse.urlencode(
            {
                "query": query,
                "orientation": "landscape",
                "per_page": 1,
                "content_filter": "high",
            }
        )
        req = urllib_request.Request(
            url,
            headers={
                "Authorization": f"Client-ID {access_key}",
                "Accept-Version": "v1",
            },
            method="GET",
        )
        try:
            with urllib_request.urlopen(req, timeout=20) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError):
            return None

        results = raw.get("results", [])
        if not results:
            return None
        first = results[0] or {}
        urls = first.get("urls", {}) or {}
        user = first.get("user", {}) or {}
        links = user.get("links", {}) or {}
        image_url = str(urls.get("regular") or urls.get("full") or urls.get("raw") or "").strip()
        photographer = str(user.get("name") or "Unsplash photographer").strip()
        photographer_url = str(links.get("html") or "https://unsplash.com").strip()
        if not image_url:
            return None
        return {
            "query": query,
            "image_url": image_url,
            "photographer": photographer,
            "photographer_url": photographer_url,
        }

    def _analyze_reference_url(self, task: str) -> dict[str, Any] | None:
        url = _extract_url(task)
        if not url:
            return None
        inspected = self._call_tool("inspect_reference_url", {"url": url, "max_chars": 12000})
        fetched = inspected
        if not str(fetched).strip() or _tool_result_failed(fetched):
            fetched = self._call_tool("fetch_url", {"url": url, "max_chars": 12000})
        if _tool_result_failed(fetched):
            return None

        class_tokens_line = ""
        headings_block = ""
        snippet_block = ""
        for block in fetched.split("\n\n"):
            if block.startswith("Class tokens:"):
                class_tokens_line = block
            elif block.startswith("Headings:"):
                headings_block = block
            elif block.startswith("Text snippet:"):
                snippet_block = block

        class_tokens = [token.strip() for token in class_tokens_line.replace("Class tokens:", "").split(",") if token.strip()]
        classes_lower = " ".join(class_tokens).lower()
        headings = [line[2:].strip() for line in headings_block.splitlines() if line.startswith("- ")]
        snippet = snippet_block.replace("Text snippet:\n", "").strip()

        mood: list[str] = []
        layout: list[str] = []
        components: list[str] = []

        if any(token in classes_lower for token in ("dark", "black", "inverse", "night")):
            mood.append("dark, high-contrast visual tone")
        if any(token in classes_lower for token in ("hero", "fullscreen", "fullheight", "banner")):
            layout.append("full-screen or oversized hero section")
        if any(token in classes_lower for token in ("masonry", "portfolio", "grid", "gallery")):
            components.append("portfolio or masonry-style grid section")
        if any(token in classes_lower for token in ("diagonal", "shape", "overlay", "mask", "line")):
            components.append("geometric accents, overlays, or directional shapes")
        if any(token in classes_lower for token in ("agency", "creative", "studio")):
            mood.append("creative-agency presentation style")

        lower_snippet = snippet.lower()
        if "portfolio" in lower_snippet or "selected work" in lower_snippet:
            components.append("project showcase framing")
        if "creative agency" in lower_snippet or "agency" in lower_snippet:
            mood.append("agency-like brand posture")
        if "scroll" in lower_snippet or "hover" in lower_snippet:
            components.append("subtle motion or hover treatment")

        if headings:
            first_heading = headings[0]
            if len(first_heading.split()) <= 8:
                layout.append("bold concise hero headline")

        summary_lines = []
        if mood:
            summary_lines.append("Mood: " + "; ".join(dict.fromkeys(mood)))
        if layout:
            summary_lines.append("Layout: " + "; ".join(dict.fromkeys(layout)))
        if components:
            summary_lines.append("Components: " + "; ".join(dict.fromkeys(components)))
        if headings:
            summary_lines.append("Headings: " + "; ".join(headings[:5]))
        if class_tokens:
            summary_lines.append("Class cues: " + ", ".join(class_tokens[:18]))
        if "Dark page: yes" in fetched:
            summary_lines.append("Overall: dark rendered page")
        hero_heading_match = re.search(r"- heading: (.+)", fetched)
        if hero_heading_match:
            summary_lines.append("Hero: " + hero_heading_match.group(1).strip())
        hero_background_match = re.search(r"- background image: (.+)", fetched)
        if hero_background_match and hero_background_match.group(1).strip() not in {"", "(none)"}:
            summary_lines.append("Hero background uses an image treatment")

        return {
            "url": url,
            "summary": "\n".join(summary_lines).strip(),
        }

    def _generate_page_html(
        self,
        task: str,
        target: str,
        current_html: str | None = None,
        style: str = "auto",
        specialist: bool = False,
        provider: BaseProvider | None = None,
    ) -> str:
        active_provider = provider or self._provider_for_request(task)
        unsplash = self._fetch_unsplash_reference(task, target)
        reference = self._analyze_reference_url(task) if specialist else None
        direction = self._design_direction(task, target, style=style)
        system_prompt = (
            "Return only complete HTML for a premium single-file landing page. "
            "No markdown fences. No explanation. "
            "The design must feel intentional and high taste, not generic AI SaaS sludge. "
            "Avoid weak default chatgpt-style layouts, purple-on-white palettes, bland card farms, filler icons, and empty marketing copy. "
            "Use a strong visual concept, disciplined spacing, sharper hierarchy, and distinctive typography choices that still load safely from standard web fonts or tasteful fallbacks. "
            "Make the page self-contained with inline CSS and mobile responsive. "
            "Prefer fewer, better sections over many shallow ones. "
            "Each section must earn its place. "
            "If the user asks for a specific style, obey it exactly instead of reverting to a generic template. "
            "Before finishing, internally check that the result looks premium, aligned, coherent, and not like a throwaway startup template. "
            "If an Unsplash image reference is provided, use it directly as a real image rather than drawing fake placeholders."
        )
        if specialist:
            system_prompt += (
                " You are acting as a dedicated design specialist. "
                "Do not settle for serviceable. Push for art direction, balance, alignment, and stronger taste."
            )
        unsplash_brief = ""
        if unsplash:
            unsplash_brief = (
                "\n\nUnsplash image reference:\n"
                f"- search query: {unsplash['query']}\n"
                f"- hotlink image URL: {unsplash['image_url']}\n"
                f"- photographer credit: Photo by {unsplash['photographer']} on Unsplash\n"
                f"- photographer profile: {unsplash['photographer_url']}\n"
                "Use this image if it fits the brief. Keep the direct image URL intact and include tasteful visible credit somewhere on the page."
            )
        reference_brief = ""
        if reference and reference.get("summary"):
            reference_brief = (
                "\n\nReference site analysis:\n"
                f"- source URL: {reference['url']}\n"
                f"{reference['summary']}\n"
                "Translate these cues into the new design directly. "
                "Do not just make a generic premium site. "
                "Echo the reference's silhouette, pacing, contrast, and composition while still producing an original page."
            )
        if current_html:
            user_prompt = (
                f"Improve this existing page for the request: {task}\n"
                f"Keep the same file target: {target}\n\n"
                f"Design direction: {direction['name']}.\n{direction['guidance']}\n\n"
                "Improve the design quality substantially rather than making tiny cosmetic changes. "
                "Fix weak composition, bad alignment, generic rhythm, and anything that feels cheap or placeholder-like.\n\n"
                "Current HTML:\n"
                f"{current_html}"
                f"{unsplash_brief}"
                f"{reference_brief}"
            )
        else:
            user_prompt = (
                f"Create a well-designed landing page for the request: {task}\n"
                f"Use this file target as context: {target}\n\n"
                f"Design direction: {direction['name']}.\n{direction['guidance']}\n\n"
                "Aim for a page that a design-conscious human would actually keep."
                f"{unsplash_brief}"
                f"{reference_brief}"
            )
        try:
            raw = active_provider.generate(system_prompt, [{"role": "user", "content": user_prompt}]).strip()
        except ProviderError:
            raw = ""
        html = _extract_html(raw) or (raw if "<html" in raw.lower() or "<!doctype html>" in raw.lower() else "")
        return html.strip() or self._fallback_landing_page_html(self.name)

    def _execute_operator_shortcut(
        self,
        task: str,
        session: dict[str, Any] | None = None,
        skill_names: list[str] | None = None,
    ) -> dict[str, Any] | None:
        lowered = task.lower().strip()
        target = self._operator_target(task, session)
        if not target:
            return None
        is_page_task = _contains_any(lowered, ("landing page", "page", "site", "html", "browser")) or target.lower().endswith(".html")

        open_only = _contains_any(lowered, (
            "open ", "reopen", "show me", "show it", "show me it", "show me that",
            "show me the page", "show me the second phase", "open again", "launch it",
        )) and not _contains_any(lowered, ("create", "build", "make", "write", "improve", "refine", "rewrite", "redesign"))
        improve = _contains_any(lowered, (
            "improve", "second pass", "wow me", "make it better", "refine", "rewrite", "redesign", "redo",
            "take another pass", "another pass", "one more pass", "pass again",
        ))
        create = _contains_any(lowered, ("create", "build", "make", "write")) and _contains_any(
            lowered, ("landing page", "page", "site", "html", "browser")
        )

        if open_only:
            result = self._call_tool("open_target", {"target": target})
            return {
                "message": f"Opened {target} in the browser.",
                "steps": [{"tool": "open_target", "args": {"target": target}, "reason": "open the current artifact", "result": result}],
            }

        if improve:
            if is_page_task:
                result = self._call_tool(
                    "design_landing_page",
                    {"path": target, "brief": task, "mode": "improve", "style": "auto", "open_after": True},
                )
                return {
                    "message": f"Improved and opened {target}.",
                    "steps": [
                        {"tool": "design_landing_page", "args": {"path": target, "brief": task, "mode": "improve", "style": "auto", "open_after": True}, "reason": "send the page through the design specialist for a stronger second pass", "result": result},
                    ],
                }
            try:
                current_html = self._call_tool("read_file", {"path": target})
            except Exception:
                current_html = ""
            html = self._generate_page_html(task, target, current_html=current_html)
            write_result = self._call_tool("write_file", {"path": target, "content": html})
            open_result = self._call_tool("open_target", {"target": target})
            return {
                "message": f"Improved and opened {target}.",
                "steps": [
                    {"tool": "write_file", "args": {"path": target, "content": html}, "reason": "rewrite the page with a stronger second pass", "result": write_result},
                    {"tool": "open_target", "args": {"target": target}, "reason": "open the improved page", "result": open_result},
                ],
            }

        if create:
            if is_page_task:
                result = self._call_tool(
                    "design_landing_page",
                    {"path": target, "brief": task, "mode": "create", "style": "auto", "open_after": True},
                )
                return {
                    "message": f"Created and opened {target}.",
                    "steps": [
                        {"tool": "design_landing_page", "args": {"path": target, "brief": task, "mode": "create", "style": "auto", "open_after": True}, "reason": "send the page through the design specialist", "result": result},
                    ],
                }
            html = self._generate_page_html(task, target)
            write_result = self._call_tool("write_file", {"path": target, "content": html})
            open_result = self._call_tool("open_target", {"target": target})
            return {
                "message": f"Created and opened {target}.",
                "steps": [
                    {"tool": "write_file", "args": {"path": target, "content": html}, "reason": "create the requested page", "result": write_result},
                    {"tool": "open_target", "args": {"target": target}, "reason": "open the new page", "result": open_result},
                ],
            }

        return None

    # ── Agent loop ─────────────────────────────────────

    def _agent_loop(
        self,
        task: str,
        session: dict[str, Any] | None = None,
        max_steps: int = 6,
        skill_names: list[str] | None = None,
        provider: BaseProvider | None = None,
    ) -> dict[str, Any]:
        history: list[dict[str, str]] = []
        steps: list[dict[str, Any]] = []
        tool_names = self._tool_names_for_skills(skill_names)
        active_provider = provider or self._provider_for_request(task)

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
            # Inject skill guidance into prompt
            skill_guidance = ""
            for sn in (skill_names or []):
                s = self._skills.get(sn)
                if s and s.get("prompt"):
                    skill_guidance += f"[skill: {s['label']}] {s['prompt']}\n"
            if skill_guidance:
                skill_guidance = "\nActive skills:\n" + skill_guidance
            prompt = SKILL_AGENT_PROMPT.replace("{tool_schemas}", schemas) + skill_guidance

            try:
                raw = active_provider.generate(prompt, messages)
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
                if not steps:
                    # Model tried to finish without any tool calls — force a retry
                    history.append({"role": "assistant", "content": json.dumps({k: v for k, v in plan.items() if k != "kind"})})
                    history.append({"role": "user", "content": "You haven't used any tools yet. You must call a tool — describe nothing, do it."})
                    continue
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

    def _normalize_tool_args(self, tool_name: str, args: dict[str, Any], session: dict[str, Any] | None = None) -> dict[str, Any]:
        normalized = dict(args)
        if tool_name == "open_target":
            if "target" not in normalized and "path" in normalized:
                normalized["target"] = normalized.pop("path")
        if tool_name in {"build_landing_page", "design_landing_page"}:
            if "path" not in normalized:
                normalized["path"] = self._operator_target("landing page", session=session) or self._default_page_target()
            if "brief" not in normalized:
                title = str(normalized.get("title", "")).strip()
                tagline = str(normalized.get("tagline", "")).strip()
                parts = [p for p in [title, tagline] if p]
                normalized["brief"] = " | ".join(parts) if parts else "Create a premium landing page."
            if "mode" not in normalized:
                normalized["mode"] = "create"
            if tool_name == "design_landing_page" and "style" not in normalized:
                normalized["style"] = "auto"
            if "open_after" not in normalized:
                normalized["open_after"] = True
            elif isinstance(normalized["open_after"], str):
                normalized["open_after"] = normalized["open_after"].strip().lower() in ("1", "true", "yes", "y", "on")
            normalized.pop("title", None)
            normalized.pop("tagline", None)
        return normalized

    def _execute_dsml_tool_calls(self, text: str, session: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        for call in _extract_dsml_tool_calls(text):
            tool_name = call.get("tool", "")
            raw_args = call.get("args", {}) or {}
            if not isinstance(raw_args, dict):
                raw_args = {}
            args = self._normalize_tool_args(tool_name, raw_args, session=session)
            result = self._call_tool(tool_name, args)
            steps.append({
                "tool": tool_name,
                "args": args,
                "reason": "execute provider-emitted tool call",
                "result": result,
            })
        return steps

    def _format_action_reply(
        self,
        surface: str,
        tool: str,
        args: dict[str, Any],
        message: str = "Done.",
    ) -> str:
        host_note = " on this Mac" if surface == "telegram" else ""
        if tool == "open_target":
            target = str(args.get("target", "")).strip()
            return f"Opened {target}{host_note}."
        if tool in {"build_landing_page", "design_landing_page"}:
            path = str(args.get("path", "")).strip()
            mode = str(args.get("mode", "create"))
            if mode == "improve":
                return f"Improved and opened {path}{host_note}."
            return f"Created and opened {path}{host_note}."
        if tool == "write_file":
            path = str(args.get("path", "")).strip()
            return f"Written to {path}."
        if tool == "run_command":
            return "Executed on your Mac."
        if tool in {"recall_session", "save_memory", "list_memories"}:
            return message.strip() or "Done."
        return message.strip() or "Done."

    # ── Main reply ─────────────────────────────────────

    def reply(self, session_id: str, user_text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        session = self._safe_load_session(session_id)
        surface = str((metadata or {}).get("surface", "")).strip().lower()
        effective_text = self._contextualize_request(user_text, session)
        effective_skills = self._effective_skill_names(effective_text, session=session)
        active_provider = self._provider_for_request(effective_text)

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

        # Fast path: deterministic operator actions should run before the model.
        if self._is_operator_request(effective_text) or self._is_open_request(effective_text):
            direct_result = self._execute_operator_shortcut(
                effective_text,
                session=session,
                skill_names=effective_skills,
            )
            if direct_result and direct_result.get("steps"):
                agent_steps = direct_result.get("steps", [])
                last_step = agent_steps[-1] if agent_steps else {}
                last_args = last_step.get("args", {}) or {}
                last_tool = str(last_step.get("tool", ""))
                last_result = str(last_step.get("result", ""))

                if _tool_result_failed(last_result):
                    reply_content = last_result
                else:
                    reply_content = self._format_action_reply(
                        surface,
                        last_tool,
                        last_args,
                        str(direct_result.get("message", "Done.")),
                    )

                loop_filtered, loop_report = filter_response(reply_content, self._adapter, context={"query": user_text})
                if loop_report.overall_passed:
                    reply_content = loop_filtered
                consistency_score = round(loop_report.overall_score, 3)

                meta: dict[str, Any] = {
                    "provider": active_provider.name,
                    "mode": self.config.companion_mode,
                    "active_skills": list(self._active_skills),
                    "auto_skills": [name for name in effective_skills if name not in self._active_skills],
                    "consistency": consistency_score,
                    "agent_steps": agent_steps,
                }
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

        try:
            draft = active_provider.generate(self._build_prompt(effective_text, session, skill_names=effective_skills), model_messages)
        except ProviderError as exc:
            # Save user message but return error gracefully
            assistant_msg: dict[str, Any] = {
                "id": str(uuid.uuid4()), "role": "assistant", "content": f"Sorry — provider error: {exc}",
                "created_at": utc_now(), "meta": {"provider": active_provider.name, "error": str(exc)},
            }
            session["messages"].append(assistant_msg)
            self._update_meta(session, user_text)
            self._session_store.save(session)
            return assistant_msg

        reply_content = draft
        agent_steps: list[dict[str, Any]] = []
        dsml_steps = self._execute_dsml_tool_calls(draft, session=session) if "<｜｜DSML｜｜" in draft else []

        # Run character adapter on the draft
        draft_filtered, draft_report = filter_response(draft, self._adapter, context={"query": user_text})
        consistency_score = round(draft_report.overall_score, 3)
        if not draft_report.overall_passed:
            reply_content = draft_filtered  # use corrected version

        if self._is_operator_request(effective_text) or self._is_open_request(effective_text):
            loop_result: dict[str, Any] | None = None
            if dsml_steps:
                only_reads = all(step.get("tool") in {"read_file", "list_files", "search_text"} for step in dsml_steps)
                if only_reads and _contains_any(effective_text.lower(), ("improve", "refine", "redesign", "rewrite", "another pass", "premium", "better")):
                    loop_result = self._execute_operator_shortcut(effective_text, session=session, skill_names=effective_skills)
                    if loop_result is None:
                        loop_result = self._agent_loop(effective_text, session=session, skill_names=effective_skills, provider=active_provider)
                else:
                    loop_result = {"message": "Executed tool actions.", "steps": dsml_steps}
            else:
                loop_result = self._execute_operator_shortcut(effective_text, session=session, skill_names=effective_skills)
                if loop_result is None:
                    loop_result = self._agent_loop(effective_text, session=session, skill_names=effective_skills, provider=active_provider)
            if loop_result:
                agent_steps = loop_result.get("steps", [])

                if agent_steps:
                    # Model executed tools — use the result as the reply
                    last_args = agent_steps[-1].get("args", {})
                    tool = agent_steps[-1].get("tool", "")
                    result_text = str(agent_steps[-1].get("result", ""))
                    if _tool_result_failed(result_text):
                        reply_content = result_text
                    else:
                        reply_content = self._format_action_reply(
                            surface,
                            tool,
                            last_args,
                            str(loop_result.get("message", "Done.")),
                        )

                # Fallback: extract HTML from the draft if model blathered instead of working
                if not agent_steps:
                    html = _extract_html(draft)
                    if html:
                        try:
                            target = self._operator_target(effective_text, session) or self._default_page_target()
                            write_result = self._call_tool("write_file", {"path": target, "content": html})
                            open_result = self._call_tool("open_target", {"target": target})
                            agent_steps = [
                                {"tool": "write_file", "args": {"path": target}, "reason": "write generated HTML", "result": write_result},
                                {"tool": "open_target", "args": {"target": target}, "reason": "open the generated page", "result": open_result},
                            ]
                            reply_content = self._format_action_reply(
                                surface,
                                "open_target",
                                {"target": target},
                                f"Created and opened {target}.",
                            )
                        except ToolError as exc:
                            agent_steps = [{"tool": "write_file", "args": {"path": target}, "result": str(exc)}]
                    else:
                        reply_content = loop_result.get("message", "Done.")

        if surface == "telegram" and not agent_steps and _looks_like_code_dump(reply_content):
            html = _extract_html(reply_content)
            if html:
                target = self._operator_target(effective_text, session) or self._default_page_target()
                write_result = self._call_tool("write_file", {"path": target, "content": html})
                open_result = self._call_tool("open_target", {"target": target})
                agent_steps = [
                    {"tool": "write_file", "args": {"path": target}, "reason": "write generated HTML", "result": write_result},
                    {"tool": "open_target", "args": {"target": target}, "reason": "open the generated page", "result": open_result},
                ]
                reply_content = self._format_action_reply(
                    surface,
                    "open_target",
                    {"target": target},
                    f"Created and opened {target}.",
                )
            else:
                reply_content = "I generated code there when I should have handled it as an action. Ask me again and I'll write it to a file instead of dumping it into Telegram."

                # Run character adapter on agent loop result
                if reply_content:
                    loop_filtered, loop_report = filter_response(reply_content, self._adapter, context={"query": user_text})
                    if loop_report.overall_passed:
                        reply_content = loop_filtered
                    consistency_score = round(loop_report.overall_score, 3)

        meta: dict[str, Any] = {
            "provider": active_provider.name,
            "mode": self.config.companion_mode,
            "active_skills": list(self._active_skills),
            "auto_skills": [name for name in effective_skills if name not in self._active_skills],
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
