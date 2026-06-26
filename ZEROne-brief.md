# ZEROne — Persona Runtime

## What it is

A portable agent runtime that enforces a consistent character/persona across any LLM provider. The key differentiator: you define the identity once in a JSON spec file, and the character adapter rewrites every model response to match — so the agent talks like itself regardless of which model is driving it.

## Location

    /Users/tgo/Documents/Playground/mip-harness/

## Files (4 total)

| File | Lines | Purpose |
|------|-------|---------|
| `zerone.py` | 871 | Core: session store, memory store, tool registry, agent loop, prompt building, operator request detection, skill system. Single class `ZEROne` that ties everything together. |
| `cli.py` | 324 | Dark-theme CLI with thinking indicator, step display, commands (`/mode`, `/provider`, `/skill`, `/session`, `/remember`, `/forget`, `/history`, etc.) |
| `providers.py` | 180 | Three providers: `OpenAIProvider`, `DeepSeekProvider`, `OllamaProvider`. All share `BaseProvider` with retry logic (3 attempts, backoff). |
| `character_adapter.py` | 1037 | The M.I.P. identity enforcement layer. Loads `persona.zeron.spec.json`, runs every response through: schema check → style enforcement → content correction (third-person→first-person, formality, emoji) → consistency scoring. |

## Key files in root

| File | Purpose |
|------|---------|
| `persona.zeron.spec.json` | ZEROne's identity definition (name, essence, voice register, formality, emoji usage) |
| `prompt-template.txt` | System prompt template — variables are filled by `_build_prompt()` |
| `setup.py` | Package setup with `zerone` console entry point |
| `README.md` | Install/usage docs |
| `skills/` | JSON skill files — drop a `.json` in, `/skill +name` to activate |
| `tests/` | 65 passing tests across 3 test files |

## How the agent loop works

1. `reply(session_id, text)` receives user input
2. If it looks like an operator request (has action+target words like "create a page"), triggers the agent loop
3. Agent loop builds a JSON-only prompt, asks the model to return `{"kind": "tool"|"final", "tool": "...", "args": {...}}`
4. Tools (`write_file`, `read_file`, `replace_in_file`, `open_target`, `shell`, `create_folder`, `list_dir`, `remember`) execute and results feed back to the model
5. When model returns `{"kind": "final"}`, loop exits and response is passed through the character adapter
6. Character adapter's `filter_response()` rewrites third-person→first-person, enforces formality, scores consistency
7. Session memory summary is saved every 5 turns for cross-session context

## What to improve

- The agent loop JSON protocol is fragile — model often returns bad JSON or omits keys
- The "Operator reached step limit" path could catch more gracefully
- Could add a `/help` command listing all available tools/showing the tool schema
- Ready to wire up Telegram but CLI should feel solid first

## Identity context

Gary built this. He's a UK-based trader/writer/builder in his 50s documenting the AI-agent era. He values directness, simplicity, and authenticity. No hype, no corporate fluff. He wants ZEROne to feel as polished as Pi, with the character adapter being the unique advantage.
