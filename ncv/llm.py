"""LLM adapter interfaces.

This module is import-safe: no API clients are created and no network calls are
made unless `generate` is invoked on a concrete remote client.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from .cache import LLMCache
from .config import NCVConfig


Message = Mapping[str, str]


class LLMClient:
    """Base interface for model providers."""

    provider: str = "base"

    def generate(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: int,
    ) -> str:
        raise NotImplementedError

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        """Backward-compatible convenience wrapper used by older package code."""

        return self.generate(
            [dict(message) for message in messages],
            model="default",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=30,
        )


ChatClient = LLMClient


@dataclass(frozen=True, slots=True)
class MockLLMClient(LLMClient):
    """Deterministic test client with optional prompt-substring fixtures."""

    response: str = ""
    fixtures: Mapping[str, str] = field(default_factory=dict)
    provider: str = "mock"

    def generate(
        self,
        messages: list[dict],
        *,
        model: str = "mock",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: int = 30,
    ) -> str:
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        for needle, fixture in self.fixtures.items():
            if needle in prompt:
                return fixture
        if self.response:
            return self.response
        return json.dumps(
            {
                "provider": self.provider,
                "model": model,
                "prompt_chars": len(prompt),
                "message_count": len(messages),
            },
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class OpenAIClient(LLMClient):
    """OpenAI chat client using the modern SDK style."""

    api_key: Optional[str] = None
    provider: str = "openai"

    def generate(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: int,
    ) -> str:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIClient; no hardcoded API key is used.")

        from openai import OpenAI

        client = OpenAI(api_key=key, timeout=timeout_s)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


@dataclass(frozen=True, slots=True)
class LocalLLMClient(LLMClient):
    """Ollama/OpenAI-compatible local chat client."""

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    provider: str = "local"

    def generate(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: int,
    ) -> str:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Local LLM provider requires the `openai` Python package for the "
                "OpenAI-compatible client. Install dependencies with `python -m pip install -r requirements.txt`."
            ) from exc

        try:
            client = OpenAI(base_url=self.base_url, api_key=self.api_key or "ollama", timeout=timeout_s)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                "Local LLM provider request failed at "
                f"{self.base_url} for model {model}. Confirm the server is running, "
                "the model is installed, and enough memory is available. "
                f"Original error: {exc}"
            ) from exc
        return response.choices[0].message.content or ""


def get_llm_client(provider: str | None, config: NCVConfig) -> LLMClient:
    """Create an uncached provider client. Callers wrap real clients with `LLMCache`."""

    selected = (provider or config.llm_provider).strip().lower()
    if selected == "mock":
        return MockLLMClient()
    if selected == "local":
        return LocalLLMClient(
            base_url=config.local_llm_base_url,
            api_key=config.local_llm_api_key,
        )
    if selected == "openai":
        return OpenAIClient()
    raise ValueError(f"Unsupported LLM provider `{provider}`; expected mock, local, or openai")


@dataclass(slots=True)
class CachedLLMClient(LLMClient):
    """Cache wrapper for successful generations."""

    client: LLMClient
    cache: LLMCache

    @property
    def provider(self) -> str:
        return getattr(self.client, "provider", self.client.__class__.__name__.lower())

    def generate(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: int,
    ) -> str:
        return self.cache.get_or_generate(
            provider=self.provider,
            model=model,
            temperature=temperature,
            messages=messages,
            generate_fn=lambda: self.client.generate(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            ),
        )


class StaticChatClient(MockLLMClient):
    """Backward-compatible alias for earlier tests."""


@dataclass(frozen=True, slots=True)
class OpenAIChatClient(OpenAIClient):
    """Backward-compatible OpenAI adapter with a default model for `complete`."""

    model: str = ""

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        model = self.model or os.environ.get("OPENAI_MODEL", "")
        if not model:
            raise RuntimeError("OPENAI_MODEL is required for OpenAIChatClient.complete")
        return self.generate(
            [dict(message) for message in messages],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=30,
        )
