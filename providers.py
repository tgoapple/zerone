"""Model providers for ZEROne — with retry logic and rate-limit handling."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any
from urllib import error, request as urllib_request


class ProviderError(RuntimeError):
    """Raised when a model provider cannot return a reply."""

MAX_RETRIES: int = 3
RETRY_DELAY: float = 1.5


def _retry_generate(fn: Any, label: str) -> str:
    """Call fn() with retries on transient errors (429, 5xx, timeouts)."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except ProviderError as exc:
            # Re-raise non-retryable errors immediately (auth, bad request, etc.)
            msg = str(exc)
            if any(code in msg for code in ("401", "403", "400")):
                raise
            if attempt == MAX_RETRIES:
                raise
            last_exc = exc
            time.sleep(RETRY_DELAY * attempt)
    raise ProviderError(f"{label} failed after {MAX_RETRIES} attempts: {last_exc}") from last_exc


class BaseProvider(ABC):
    name: str
    model: str | None = None

    @abstractmethod
    def generate(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


# ── Conversation helpers ────────────────────────────────────


def build_chat_body(system_prompt: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    body: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for m in messages:
        body.append({"role": m.get("role", "user"), "content": m["content"]})
    return body


# ── Real providers ──────────────────────────────────────────


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("MIP_OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MIP_OPENAI_API_KEY") or ""

    def _call(self, body: bytes) -> dict[str, Any]:
        req = urllib_request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib_request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode())

    def generate(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is not configured.")
        payload = json.dumps({"model": self.model, "messages": build_chat_body(system_prompt, messages)}).encode()

        def attempt() -> str:
            try:
                raw = self._call(payload)
            except error.HTTPError as exc:
                raise ProviderError(f"OpenAI request failed: {exc.code} {exc.read().decode(errors='replace')}") from exc
            except error.URLError as exc:
                raise ProviderError(f"OpenAI request failed: {exc.reason}") from exc
            try:
                return str(raw["choices"][0]["message"]["content"]).strip()
            except (KeyError, TypeError, IndexError) as exc:
                raise ProviderError("OpenAI response missing content.") from exc

        return _retry_generate(attempt, "OpenAI")


class DeepSeekProvider(BaseProvider):
    name = "deepseek"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("MIP_DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MIP_DEEPSEEK_API_KEY") or ""

    def _call(self, body: bytes) -> dict[str, Any]:
        req = urllib_request.Request(
            "https://api.deepseek.com/chat/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib_request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())

    def generate(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise ProviderError("DEEPSEEK_API_KEY is not configured.")
        payload = json.dumps({"model": self.model, "messages": build_chat_body(system_prompt, messages)}).encode()

        def attempt() -> str:
            try:
                raw = self._call(payload)
            except error.HTTPError as exc:
                raise ProviderError(f"DeepSeek request failed: {exc.code} {exc.read().decode(errors='replace')}") from exc
            except error.URLError as exc:
                raise ProviderError(f"DeepSeek request failed: {exc.reason}") from exc
            try:
                return str(raw["choices"][0]["message"]["content"]).strip()
            except (KeyError, TypeError, IndexError) as exc:
                raise ProviderError("DeepSeek response missing content.") from exc

        return _retry_generate(attempt, "DeepSeek")


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or "qwen2.5:7b"
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def _call(self, body: bytes) -> dict[str, Any]:
        req = urllib_request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib_request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())

    def generate(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        payload = json.dumps({"model": self.model, "messages": build_chat_body(system_prompt, messages)}).encode()

        def attempt() -> str:
            try:
                raw = self._call(payload)
            except error.HTTPError as exc:
                raise ProviderError(f"Ollama request failed: {exc.code} {exc.read().decode(errors='replace')}") from exc
            except error.URLError as exc:
                raise ProviderError(f"Ollama request failed: {exc.reason}") from exc
            try:
                return str(raw["message"]["content"]).strip()
            except (KeyError, TypeError) as exc:
                raise ProviderError("Ollama response missing content.") from exc

        return _retry_generate(attempt, "Ollama")


# ── Factory ────────────────────────────────────────────────


def build_provider(name: str, model: str | None = None) -> BaseProvider:
    normalized = (name or "deepseek").strip().lower()
    if normalized == "openai":
        return OpenAIProvider(model=model)
    if normalized == "deepseek":
        return DeepSeekProvider(model=model)
    if normalized == "ollama":
        return OllamaProvider(model=model)
    raise ProviderError(f"Unknown provider: {normalized}. Valid options: openai, deepseek, ollama.")
