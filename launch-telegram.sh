#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -f "$ROOT_DIR/.env.local" ]]; then
  # Load local runtime secrets without committing them.
  set -a
  source "$ROOT_DIR/.env.local"
  set +a
fi

echo
echo "ZEROne"
echo "Telegram companion + operator"
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required."
  exit 1
fi

if [[ ! -f "$ROOT_DIR/telegram_bot.py" ]]; then
  echo "telegram_bot.py not found."
  exit 1
fi

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
  echo "TELEGRAM_BOT_TOKEN is not set."
  echo
  read -r "?Paste your Telegram bot token: " TELEGRAM_BOT_TOKEN
  export TELEGRAM_BOT_TOKEN
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo
  echo "No provider API key found."
  echo "Set DEEPSEEK_API_KEY or OPENAI_API_KEY before continuing."
  exit 1
fi

PROVIDER="${MIP_PROVIDER:-deepseek}"
WORKSPACE="${MIP_WORKSPACE_ROOT:-$ROOT_DIR}"

echo
echo "Starting Telegram bridge..."
echo "provider: $PROVIDER"
echo "workspace: $WORKSPACE"
echo

exec python3 "$ROOT_DIR/telegram_bot.py" --provider "$PROVIDER" --workspace "$WORKSPACE"
