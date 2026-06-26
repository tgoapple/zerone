# ZEROne — M.I.P. Runtime

ZEROne is a portable character runtime for LLM agents. The core idea: define a persona once in a spec file, and the **character adapter** enforces it on every model response — regardless of which provider is underneath.

Switch from DeepSeek to OpenAI to Ollama. The character stays the same.

## What makes it different

- **Character adapter pipeline** — every model response goes through schema checking, style enforcement, content correction (third-person → first-person, verb conjugation), injection detection, and consistency scoring. The prompt steers, the adapter guarantees.
- **Skills as JSON files** — drop a `.json` into `skills/` and it's available. Activate with `/skill +<name>`.
- **Tool registry** — tools are registered with schemas and handlers. Adding a new tool means registering a function.
- **Session memory compounding** — key context carries across sessions automatically.
- **Retry with backoff** — transient API errors (429, 5xx) are retried up to 3 times.
- **Context overflow protection** — long sessions are trimmed automatically.
- **Error handling** on every path — bad JSON from model, API errors, corrupt session files, missing persona files all produce graceful messages.

## Quick start

```bash
# Run with DeepSeek (default)
python3 cli.py

# Run with a specific provider and model
python3 cli.py --provider openai --model gpt-4o-mini
python3 cli.py --provider ollama --model qwen2.5:7b
python3 cli.py --provider deepseek --model deepseek-v4-flash

# One-shot reply and exit
python3 cli.py --one-shot "Who are you?"

# Custom session
python3 cli.py --session my-project

# Telegram bridge
export TELEGRAM_BOT_TOKEN=123456:abc...
./launch-telegram.sh
```

## Telegram

ZEROne can also run as a lean Telegram surface using the same runtime as the CLI.

```bash
export TELEGRAM_BOT_TOKEN=123456:abc...
export DEEPSEEK_API_KEY=sk-...
./launch-telegram.sh
```

What it does:

- Uses the same ZEROne core, memory, and workspace tools as the CLI
- Shows Telegram typing presence while ZEROne is working
- Splits long replies safely so Telegram does not truncate them
- Supports `/help` and `/status`

## Commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/status` | Runtime info (provider, model, session, mode) |
| `/history` | Last 8 turns of conversation |
| `/mode <name>` | Switch mode: companion, operator, teacher, brainstorm, reflect |
| `/provider <name>` | Switch provider: deepseek, openai, ollama |
| `/model <name>` | Switch model |
| `/skills` | List available skills |
| `/skill +<name>` | Activate a skill |
| `/skill -<name>` | Deactivate a skill |
| `/skill reload` | Reload skills from disk |
| `/memories` | List stored memories |
| `/remember <text>` | Save a fact |
| `/forget <id>` | Remove a memory |
| `/sessions` | List sessions |
| `/session <name>` | Switch to a session |
| `/new <name>` | Start a fresh session |
| `/exit` | Quit |

## Skills

Skills are JSON files in the `skills/` directory. Example:

```json
{
  "name": "web-dev",
  "label": "Web Development",
  "prompt": "You have deep knowledge of HTML, CSS, JavaScript, and modern web frameworks.",
  "tools": ["read_file", "write_file", "replace_in_file", "open_target", "run_command"]
}
```

Activate with `/skill +web-dev`.

## Environment variables

```
# Provider keys
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=sk-...

# Optional overrides
MIP_DEEPSEEK_MODEL=deepseek-chat
MIP_OPENAI_MODEL=gpt-4o-mini
MIP_WORKSPACE_ROOT=/path/to/workspace

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
```

## Architecture

```
cli.py / telegram_bot.py
    └── zerone.py (ZEROne class)
            ├── providers.py (OpenAI, DeepSeek, Ollama with retry)
            ├── character_adapter.py (persona enforcement pipeline)
            ├── persona.zeron.spec.json (persona definition)
            ├── prompt-template.txt (system prompt template)
            └── skills/ (JSON skill definitions)
```

## Files

| File | Lines | Purpose |
|---|---|---|
| `zerone.py` | ~830 | Core runtime: session store, memory store, tool registry, agent loop, reply pipeline, skill system |
| `cli.py` | ~320 | Terminal interface with dark theme, thinking indicator |
| `providers.py` | ~210 | Provider wrappers with retry logic |
| `character_adapter.py` | ~1037 | Persona enforcement: schema check, style enforcement, content correction, injection detection, scoring |

Total: ~2400 lines across 4 files.
