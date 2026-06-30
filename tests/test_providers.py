"""Tests for ZEROne providers."""
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from providers import (
    CodexProvider, OpenAIProvider, DeepSeekProvider, OllamaProvider,
    build_provider, ProviderError,
)


class TestBuildProvider:
    def test_build_openai(self):
        p = build_provider("openai")
        assert isinstance(p, OpenAIProvider)

    def test_build_deepseek(self):
        p = build_provider("deepseek")
        assert isinstance(p, DeepSeekProvider)

    def test_build_ollama(self):
        p = build_provider("ollama")
        assert isinstance(p, OllamaProvider)

    def test_build_codex(self):
        p = build_provider("codex")
        assert isinstance(p, CodexProvider)

    def test_build_unknown(self):
        with pytest.raises(ProviderError):
            build_provider("nonexistent")

    def test_build_models(self):
        p = build_provider("openai", "gpt-4")
        assert p.model == "gpt-4"


class TestProviderErrors:
    def test_openai_no_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            p = OpenAIProvider()
            with pytest.raises(ProviderError, match="not configured"):
                p.generate("prompt", [{"role": "user", "content": "hi"}])

    def test_deepseek_no_key(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            p = DeepSeekProvider()
            with pytest.raises(ProviderError, match="not configured"):
                p.generate("prompt", [{"role": "user", "content": "hi"}])

    def test_ollama_local(self):
        """Ollama doesn't require a key, but will fail locally."""
        p = OllamaProvider(base_url="http://localhost:99999")
        with pytest.raises(ProviderError):
            p.generate("prompt", [{"role": "user", "content": "hi"}])

    def test_codex_missing_cli(self):
        p = CodexProvider(binary="missing-codex")
        with patch("providers.shutil.which", return_value=None):
            with pytest.raises(ProviderError, match="not installed"):
                p.generate("prompt", [{"role": "user", "content": "hi"}])

    def test_codex_reads_last_message_file(self):
        p = CodexProvider(model="gpt-5.4", binary="codex")

        def fake_run(args, **kwargs):
            output_path = Path(args[args.index("--output-last-message") + 1])
            output_path.write_text("hello from codex", encoding="utf-8")
            return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("providers.shutil.which", return_value="/usr/local/bin/codex"):
            with patch("providers.subprocess.run", side_effect=fake_run):
                result = p.generate("prompt", [{"role": "user", "content": "hi"}])
        assert result == "hello from codex"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
