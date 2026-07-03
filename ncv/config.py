"""Central configuration for the NCV pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class NCVConfig:
    """Runtime configuration shared across ingestion, retrieval, and verification."""

    books_dir: Path = Path("Dataset/Books")
    train_path: Path = Path("Dataset/train.csv")
    test_path: Path = Path("Dataset/test.csv")
    output_dir: Path = Path("outputs")
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 400
    chunk_overlap: int = 100
    top_k_retrieval: int = 10
    llm_provider: str = "local"
    max_claims: int = 6
    strict_character_filter: bool = False
    use_embeddings: bool = False
    embedding_mode: str = "auto"
    use_llm_cache: bool = True
    llm_cache_path: Path = Path(".cache/llm_cache.jsonl")
    local_llm_model: str = "gemma4"
    local_llm_base_url: str = "http://localhost:11434/v1"
    local_llm_api_key: str = "ollama"
    openai_model: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 700
    llm_timeout_s: int = 120
    use_lexical: bool = True
    use_character_scoring: bool = True
    use_event_scoring: bool = True
    use_contradiction_families: bool = True
    use_timeline_features: bool = True
    use_rerank: bool = True
    use_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "books_dir", Path(self.books_dir))
        object.__setattr__(self, "train_path", Path(self.train_path))
        object.__setattr__(self, "test_path", Path(self.test_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "llm_cache_path", Path(self.llm_cache_path))

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.top_k_retrieval <= 0:
            raise ValueError("top_k_retrieval must be positive")
        if self.max_claims <= 0:
            raise ValueError("max_claims must be positive")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model is required")
        if not self.llm_provider.strip():
            raise ValueError("llm_provider is required")
        if not self.local_llm_model.strip():
            raise ValueError("local_llm_model is required")
        if not self.local_llm_base_url.strip():
            raise ValueError("local_llm_base_url is required")
        if self.llm_temperature < 0:
            raise ValueError("llm_temperature cannot be negative")
        if self.llm_max_tokens <= 0:
            raise ValueError("llm_max_tokens must be positive")
        if self.llm_timeout_s <= 0:
            raise ValueError("llm_timeout_s must be positive")
        if not self.embedding_mode.strip():
            raise ValueError("embedding_mode is required")
        if not self.cross_encoder_model.strip():
            raise ValueError("cross_encoder_model is required")

    @classmethod
    def from_env(cls, prefix: str = "NCV_") -> "NCVConfig":
        """Build config from environment variables, falling back to defaults."""

        dotenv = _read_dotenv(Path(".env"))
        defaults = cls()

        def env_value(name: str) -> str | None:
            return os.environ.get(name) or os.environ.get(prefix + name) or dotenv.get(name) or dotenv.get(prefix + name)

        def get_path(name: str, default: Path) -> Path:
            return Path(env_value(name) or str(default))

        def get_str(name: str, default: str) -> str:
            return env_value(name) or default

        def get_int(name: str, default: int) -> int:
            raw = env_value(name)
            return default if raw is None else int(raw)

        def get_float(name: str, default: float) -> float:
            raw = env_value(name)
            return default if raw is None else float(raw)

        def get_bool(name: str, default: bool) -> bool:
            raw = env_value(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            books_dir=get_path("BOOKS_DIR", defaults.books_dir),
            train_path=get_path("TRAIN_PATH", defaults.train_path),
            test_path=get_path("TEST_PATH", defaults.test_path),
            output_dir=get_path("OUTPUT_DIR", defaults.output_dir),
            embedding_model=get_str("EMBEDDING_MODEL", defaults.embedding_model),
            chunk_size=get_int("CHUNK_SIZE", defaults.chunk_size),
            chunk_overlap=get_int("CHUNK_OVERLAP", defaults.chunk_overlap),
            top_k_retrieval=get_int("TOP_K_RETRIEVAL", defaults.top_k_retrieval),
            llm_provider=get_str("LLM_PROVIDER", defaults.llm_provider),
            max_claims=get_int("MAX_CLAIMS", defaults.max_claims),
            strict_character_filter=get_bool("STRICT_CHARACTER_FILTER", defaults.strict_character_filter),
            use_embeddings=get_bool("USE_EMBEDDINGS", defaults.use_embeddings),
            embedding_mode=get_str("EMBEDDING_MODE", defaults.embedding_mode),
            use_llm_cache=get_bool("USE_CACHE", get_bool("USE_LLM_CACHE", defaults.use_llm_cache)),
            llm_cache_path=get_path("LLM_CACHE_PATH", defaults.llm_cache_path),
            local_llm_model=get_str("LOCAL_LLM_MODEL", defaults.local_llm_model),
            local_llm_base_url=get_str("LOCAL_LLM_BASE_URL", defaults.local_llm_base_url),
            local_llm_api_key=get_str("LOCAL_LLM_API_KEY", defaults.local_llm_api_key),
            openai_model=get_str("OPENAI_MODEL", defaults.openai_model),
            llm_temperature=get_float("LLM_TEMPERATURE", defaults.llm_temperature),
            llm_max_tokens=get_int("LLM_MAX_TOKENS", defaults.llm_max_tokens),
            llm_timeout_s=get_int("LLM_TIMEOUT_S", defaults.llm_timeout_s),
            use_lexical=get_bool("USE_LEXICAL", defaults.use_lexical),
            use_character_scoring=get_bool("USE_CHARACTER_SCORING", defaults.use_character_scoring),
            use_event_scoring=get_bool("USE_EVENT_SCORING", defaults.use_event_scoring),
            use_contradiction_families=get_bool("USE_CONTRADICTION_FAMILIES", defaults.use_contradiction_families),
            use_timeline_features=get_bool("USE_TIMELINE_FEATURES", defaults.use_timeline_features),
            use_rerank=get_bool("USE_RERANK", defaults.use_rerank),
            use_cross_encoder=get_bool("USE_CROSS_ENCODER", defaults.use_cross_encoder),
            cross_encoder_model=get_str("CROSS_ENCODER_MODEL", defaults.cross_encoder_model),
        )

    def ensure_output_dir(self) -> Path:
        """Create and return the configured output directory."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def model_for_provider(self, provider: str | None = None) -> str:
        selected = (provider or self.llm_provider).strip().lower()
        if selected == "local":
            return self.local_llm_model
        if selected == "openai":
            if not self.openai_model.strip():
                raise ValueError("OPENAI_MODEL must be set when LLM_PROVIDER=openai")
            return self.openai_model
        if selected == "mock":
            return "mock"
        raise ValueError(f"Unsupported LLM provider: {selected}")


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read a simple `.env` file without adding a runtime dependency."""

    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


DEFAULT_CONFIG = NCVConfig()
