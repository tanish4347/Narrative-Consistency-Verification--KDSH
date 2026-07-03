"""Report what is visible from the local LLM cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the NCV LLM cache.")
    parser.add_argument("--cache", default=".cache/llm_cache.jsonl")
    parser.add_argument("--output", default="outputs/cache_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_cache_report(args.cache)
    output_path = save_cache_report(report, args.output)
    print(format_cache_report(report))
    print(f"Wrote cache report to {output_path}")


def build_cache_report(path: str | Path) -> dict[str, object]:
    cache_path = Path(path)
    entries = _load_entries(cache_path)
    keys = [str(entry.get("key", "")) for entry in entries if entry.get("key")]
    unique_keys = set(keys)
    providers = sorted({str(entry.get("provider", "")) for entry in entries if entry.get("provider")})
    models = sorted({str(entry.get("model", "")) for entry in entries if entry.get("model")})
    return {
        "cache_file_path": str(cache_path),
        "cache_exists": cache_path.exists(),
        "cached_llm_calls": len(entries),
        "unique_prompts": len(unique_keys),
        "providers_seen": providers,
        "models_seen": models,
        "estimated_calls_avoided": max(len(keys) - len(unique_keys), 0),
        "note": "Tests and smoke runs use mock mode. Exact cache hits require runtime CACHE_HIT logs; duplicate cache keys provide only a lower-bound estimate of calls avoided.",
    }


def save_cache_report(report: dict[str, object], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_cache_report(report), encoding="utf-8")
    return output_path


def format_cache_report(report: dict[str, object]) -> str:
    lines = [
        "# Cache Report",
        "",
        f"- Cache file path: `{report['cache_file_path']}`",
        f"- Cache exists: {report['cache_exists']}",
        f"- Number of cached LLM calls: {report['cached_llm_calls']}",
        f"- Unique prompts: {report['unique_prompts']}",
        f"- Providers/models seen: {_providers_models(report)}",
        f"- Estimated calls avoided: {report['estimated_calls_avoided']}",
        f"- Note: {report['note']}",
        "",
    ]
    return "\n".join(lines)


def _load_entries(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    entries: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def _providers_models(report: dict[str, object]) -> str:
    providers = report.get("providers_seen") or []
    models = report.get("models_seen") or []
    provider_text = ", ".join(str(item) for item in providers) or "none"
    model_text = ", ".join(str(item) for item in models) or "none"
    return f"providers={provider_text}; models={model_text}"


if __name__ == "__main__":
    main()
