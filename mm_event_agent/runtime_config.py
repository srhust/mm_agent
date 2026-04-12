"""Centralized runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    event_type_mode: str
    debug: bool
    log_level: str

    openai_api_key: str
    openai_model: str
    openai_base_url: str
    openai_timeout_seconds: float

    tavily_api_key: str
    tavily_endpoint: str
    search_timeout_seconds: float
    search_max_results: int
    search_top_k: int
    search_min_relevance: float

    florence2_model_id: str
    florence2_task: str
    florence2_device: str

    florence2_local_endpoint: str
    florence2_local_timeout_seconds: float


def load_settings() -> Settings:
    return Settings(
        event_type_mode=_env_str("MM_EVENT_TYPE_MODE", "closed_set") or "closed_set",
        debug=_env_flag("MM_EVENT_DEBUG", False),
        log_level=_env_str("MM_AGENT_LOG_LEVEL", "INFO") or "INFO",
        openai_api_key=_env_str("OPENAI_API_KEY", ""),
        openai_model=_env_str("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        openai_base_url=_env_str("OPENAI_BASE_URL", ""),
        openai_timeout_seconds=_env_float("OPENAI_TIMEOUT_SECONDS", 30.0),
        tavily_api_key=_env_str("TAVILY_API_KEY", ""),
        tavily_endpoint=_env_str("MM_EVENT_TAVILY_ENDPOINT", "https://api.tavily.com/search") or "https://api.tavily.com/search",
        search_timeout_seconds=_env_float("MM_EVENT_SEARCH_TIMEOUT_SECONDS", 8.0),
        search_max_results=max(1, _env_int("MM_EVENT_SEARCH_MAX_RESULTS", 5)),
        search_top_k=max(1, _env_int("MM_EVENT_SEARCH_TOP_K", 3)),
        search_min_relevance=max(0.0, min(1.0, _env_float("MM_EVENT_SEARCH_MIN_RELEVANCE", 0.18))),
        florence2_model_id=_env_str("FLORENCE2_MODEL_ID", "microsoft/Florence-2-base-ft") or "microsoft/Florence-2-base-ft",
        florence2_task=_env_str("FLORENCE2_TASK", "<OPEN_VOCABULARY_DETECTION>") or "<OPEN_VOCABULARY_DETECTION>",
        florence2_device=_env_str("FLORENCE2_DEVICE", ""),
        florence2_local_endpoint=_env_str("FLORENCE2_LOCAL_ENDPOINT", ""),
        florence2_local_timeout_seconds=_env_float("FLORENCE2_LOCAL_TIMEOUT_SECONDS", 10.0),
    )


settings = load_settings()
