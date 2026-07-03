"""Small disk cache utilities for expensive model calls."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .utils import stable_hash


logger = logging.getLogger("ncv.cache")


class JsonDiskCache:
    """Hash-addressed JSON cache stored as one file per payload."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def make_key(payload: Mapping[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[dict[str, Any]]:
        path = self.path_for_key(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: Mapping[str, Any]) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for_key(key)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def get_or_set(self, payload: Mapping[str, Any], value: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        key = self.make_key(payload)
        cached = self.get(key)
        if cached is not None:
            return key, cached
        self.set(key, value)
        return key, dict(value)


class LLMCache:
    """Append-only JSONL cache for successful LLM generations."""

    def __init__(self, path: str | Path = ".cache/llm_cache.jsonl", *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._entries: dict[str, dict[str, Any]] = {}
        if self.enabled:
            self._load()

    @staticmethod
    def make_key(
        *,
        provider: str,
        model: str,
        temperature: float,
        messages: Sequence[Mapping[str, str]],
    ) -> str:
        return stable_hash(
            {
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "messages": list(messages),
            }
        )

    def get(self, key: str) -> Optional[str]:
        if not self.enabled:
            logger.info("CACHE_MISS disabled %s", key)
            return None
        entry = self._entries.get(key)
        if entry is None:
            logger.info("CACHE_MISS %s", key)
            return None
        logger.info("CACHE_HIT %s", key)
        return str(entry["output"])

    def set(
        self,
        key: str,
        *,
        provider: str,
        model: str,
        temperature: float,
        messages: Sequence[Mapping[str, str]],
        output: str,
    ) -> None:
        if not self.enabled:
            return
        entry = {
            "key": key,
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "messages": list(messages),
            "output": output,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        self._entries[key] = entry

    def get_or_generate(
        self,
        *,
        provider: str,
        model: str,
        temperature: float,
        messages: Sequence[Mapping[str, str]],
        generate_fn,
    ) -> str:
        key = self.make_key(provider=provider, model=model, temperature=temperature, messages=messages)
        cached = self.get(key)
        if cached is not None:
            return cached
        output = generate_fn()
        self.set(
            key,
            provider=provider,
            model=model,
            temperature=temperature,
            messages=messages,
            output=output,
        )
        return output

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed cache line in %s", self.path)
                    continue
                key = entry.get("key")
                if key and "output" in entry:
                    self._entries[str(key)] = entry
