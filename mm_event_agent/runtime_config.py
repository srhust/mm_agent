"""Centralized runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os

VALID_RUN_MODES = frozenset({"benchmark", "open_world"})


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


def _normalize_run_mode(value: str, default: str = "benchmark") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in VALID_RUN_MODES:
        return normalized
    return default


def _effective_search_enabled(run_mode: str, enable_search: bool) -> bool:
    return run_mode == "open_world" and bool(enable_search)


@dataclass(frozen=True)
class Settings:
    event_type_mode: str
    debug: bool
    log_level: str
    run_mode: str
    enable_search: bool

    openai_api_key: str
    openai_model: str
    openai_base_url: str
    openai_timeout_seconds: float
    extraction_backend: str
    extraction_model_name: str
    extraction_api_base: str
    extraction_api_key: str
    extraction_timeout_seconds: float
    extraction_stage_a_use_raw_image: bool

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
    use_vlm_perception: bool
    perception_backend: str
    perception_model_name: str
    perception_api_base: str
    perception_api_key: str
    perception_timeout_seconds: float
    perception_instruction: str
    perception_cache_enabled: bool

    rag_use_persistent_index: bool
    rag_use_demo_corpus: bool
    rag_index_root: str
    rag_qwen_model_path: str
    rag_qwen_device: str
    rag_qwen_dtype: str
    rag_qwen_attn_impl: str
    rag_text_encoder_model: str
    rag_text_encoder_model_path: str
    rag_qwen_embedding_model_path: str
    rag_qwen_embedding_device: str
    rag_qwen_embedding_dtype: str
    rag_qwen_embedding_attn_impl: str
    rag_qwen_text_instruction: str
    rag_qwen_image_instruction: str
    rag_qwen_embedding_out_dim: int
    rag_qwen_embedding_normalize: bool
    rag_image_encoder_model_path: str
    rag_image_encoder_device: str
    rag_ace_text_index_dir: str
    rag_maven_text_index_dir: str
    rag_swig_text_index_dir: str
    rag_swig_image_index_dir: str
    rag_bridge_index_dir: str
    rag_default_top_k: int
    rag_text_top_k: int
    rag_image_top_k: int
    rag_bridge_top_k: int
    rag_enable_image_query: bool

    @property
    def effective_search_enabled(self) -> bool:
        return _effective_search_enabled(self.run_mode, self.enable_search)


def load_settings() -> Settings:
    return Settings(
        event_type_mode=_env_str("MM_EVENT_TYPE_MODE", "closed_set") or "closed_set",
        debug=_env_flag("MM_EVENT_DEBUG", False),
        log_level=_env_str("MM_AGENT_LOG_LEVEL", "INFO") or "INFO",
        run_mode=_normalize_run_mode(_env_str("MM_EVENT_RUN_MODE", "benchmark"), default="benchmark"),
        enable_search=_env_flag("MM_EVENT_ENABLE_SEARCH", False),
        openai_api_key=_env_str("OPENAI_API_KEY", ""),
        openai_model=_env_str("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        openai_base_url=_env_str("OPENAI_BASE_URL", ""),
        openai_timeout_seconds=_env_float("OPENAI_TIMEOUT_SECONDS", 30.0),
        extraction_backend=_env_str("MM_EVENT_EXTRACTION_BACKEND", "openai_compatible") or "openai_compatible",
        extraction_model_name=_env_str(
            "MM_EVENT_EXTRACTION_MODEL_NAME",
            _env_str("OPENAI_MODEL", "gpt-4o-mini"),
        )
        or "gpt-4o-mini",
        extraction_api_base=_env_str("MM_EVENT_EXTRACTION_API_BASE", _env_str("OPENAI_BASE_URL", "")),
        extraction_api_key=_env_str("MM_EVENT_EXTRACTION_API_KEY", _env_str("OPENAI_API_KEY", "")),
        extraction_timeout_seconds=_env_float(
            "MM_EVENT_EXTRACTION_TIMEOUT_SECONDS",
            _env_float("OPENAI_TIMEOUT_SECONDS", 30.0),
        ),
        extraction_stage_a_use_raw_image=_env_flag("MM_EVENT_EXTRACTION_STAGE_A_USE_RAW_IMAGE", False),
        tavily_api_key=_env_str("TAVILY_API_KEY", ""),
        tavily_endpoint=_env_str("MM_EVENT_TAVILY_ENDPOINT", "https://api.tavily.com/search") or "https://api.tavily.com/search",
        search_timeout_seconds=_env_float("MM_EVENT_SEARCH_TIMEOUT_SECONDS", 8.0),
        search_max_results=max(1, _env_int("MM_EVENT_SEARCH_MAX_RESULTS", 5)),
        search_top_k=max(1, _env_int("MM_EVENT_SEARCH_TOP_K", 3)),
        search_min_relevance=max(0.0, min(1.0, _env_float("MM_EVENT_SEARCH_MIN_RELEVANCE", 0.18))),
        florence2_model_id=_env_str("FLORENCE2_MODEL_ID", "microsoft/Florence-2-base-ft") or "microsoft/Florence-2-base-ft",
        florence2_task=_env_str("FLORENCE2_TASK", "<CAPTION_TO_PHRASE_GROUNDING>") or "<CAPTION_TO_PHRASE_GROUNDING>",
        florence2_device=_env_str("FLORENCE2_DEVICE", ""),
        florence2_local_endpoint=_env_str("FLORENCE2_LOCAL_ENDPOINT", ""),
        florence2_local_timeout_seconds=_env_float("FLORENCE2_LOCAL_TIMEOUT_SECONDS", 10.0),
        use_vlm_perception=_env_flag("MM_EVENT_USE_VLM_PERCEPTION", False),
        perception_backend=_env_str("MM_EVENT_PERCEPTION_BACKEND", "openai_compatible") or "openai_compatible",
        perception_model_name=_env_str("MM_EVENT_PERCEPTION_MODEL_NAME", "qwen3.5-plus") or "qwen3.5-plus",
        perception_api_base=_env_str("MM_EVENT_PERCEPTION_API_BASE", _env_str("OPENAI_BASE_URL", "")),
        perception_api_key=_env_str("MM_EVENT_PERCEPTION_API_KEY", _env_str("OPENAI_API_KEY", "")),
        perception_timeout_seconds=_env_float(
            "MM_EVENT_PERCEPTION_TIMEOUT_SECONDS",
            _env_float("OPENAI_TIMEOUT_SECONDS", 30.0),
        ),
        perception_instruction=_env_str(
            "MM_EVENT_PERCEPTION_INSTRUCTION",
            (
                "You are a multimodal perception stage. Analyze the image and optional text context. "
                'Return strict JSON with keys "image_desc" and "perception_summary". '
                '"image_desc" must be a concise factual image description. '
                '"perception_summary" must be a short multimodal summary suitable for downstream event extraction.'
            ),
        ),
        perception_cache_enabled=_env_flag("MM_EVENT_PERCEPTION_CACHE_ENABLED", True),
        rag_use_persistent_index=_env_flag("MM_EVENT_RAG_USE_PERSISTENT_INDEX", False),
        rag_use_demo_corpus=_env_flag("MM_EVENT_RAG_USE_DEMO_CORPUS", True),
        rag_index_root=_env_str("MM_EVENT_RAG_INDEX_ROOT", "data/rag/indexes") or "data/rag/indexes",
        rag_qwen_model_path=_env_str(
            "MM_EVENT_RAG_QWEN_MODEL_PATH",
            _env_str("MM_EVENT_RAG_QWEN_EMBEDDING_MODEL_PATH", ""),
        ),
        rag_qwen_device=_env_str(
            "MM_EVENT_RAG_QWEN_DEVICE",
            _env_str("MM_EVENT_RAG_QWEN_EMBEDDING_DEVICE", "cuda:0"),
        )
        or "cuda:0",
        rag_qwen_dtype=_env_str(
            "MM_EVENT_RAG_QWEN_DTYPE",
            _env_str("MM_EVENT_RAG_QWEN_EMBEDDING_DTYPE", "bfloat16"),
        )
        or "bfloat16",
        rag_qwen_attn_impl=_env_str(
            "MM_EVENT_RAG_QWEN_ATTN_IMPL",
            _env_str("MM_EVENT_RAG_QWEN_EMBEDDING_ATTN_IMPL", "sdpa"),
        )
        or "sdpa",
        rag_text_encoder_model=_env_str("MM_EVENT_RAG_TEXT_ENCODER_MODEL", "sentence-transformers/all-MiniLM-L6-v2") or "sentence-transformers/all-MiniLM-L6-v2",
        rag_text_encoder_model_path=_env_str("MM_EVENT_RAG_TEXT_ENCODER_MODEL_PATH", ""),
        rag_qwen_embedding_model_path=_env_str(
            "MM_EVENT_RAG_QWEN_EMBEDDING_MODEL_PATH",
            _env_str("MM_EVENT_RAG_QWEN_MODEL_PATH", ""),
        ),
        rag_qwen_embedding_device=_env_str(
            "MM_EVENT_RAG_QWEN_EMBEDDING_DEVICE",
            _env_str("MM_EVENT_RAG_QWEN_DEVICE", "cuda:0"),
        )
        or "cuda:0",
        rag_qwen_embedding_dtype=_env_str(
            "MM_EVENT_RAG_QWEN_EMBEDDING_DTYPE",
            _env_str("MM_EVENT_RAG_QWEN_DTYPE", "bfloat16"),
        )
        or "bfloat16",
        rag_qwen_embedding_attn_impl=_env_str(
            "MM_EVENT_RAG_QWEN_EMBEDDING_ATTN_IMPL",
            _env_str("MM_EVENT_RAG_QWEN_ATTN_IMPL", "sdpa"),
        )
        or "sdpa",
        rag_qwen_text_instruction=_env_str(
            "MM_EVENT_RAG_QWEN_TEXT_INSTRUCTION",
            "Retrieve text relevant to the user's query.",
        )
        or "Retrieve text relevant to the user's query.",
        rag_qwen_image_instruction=_env_str(
            "MM_EVENT_RAG_QWEN_IMAGE_INSTRUCTION",
            "Retrieve images relevant to the user's query.",
        )
        or "Retrieve images relevant to the user's query.",
        rag_qwen_embedding_out_dim=max(0, _env_int("MM_EVENT_RAG_QWEN_EMBEDDING_OUT_DIM", 0)),
        rag_qwen_embedding_normalize=_env_flag("MM_EVENT_RAG_QWEN_EMBEDDING_NORMALIZE", True),
        rag_image_encoder_model_path=_env_str("MM_EVENT_RAG_IMAGE_ENCODER_MODEL_PATH", ""),
        rag_image_encoder_device=_env_str("MM_EVENT_RAG_IMAGE_ENCODER_DEVICE", ""),
        rag_ace_text_index_dir=_env_str("MM_EVENT_RAG_ACE_TEXT_INDEX_DIR", ""),
        rag_maven_text_index_dir=_env_str("MM_EVENT_RAG_MAVEN_TEXT_INDEX_DIR", ""),
        rag_swig_text_index_dir=_env_str("MM_EVENT_RAG_SWIG_TEXT_INDEX_DIR", ""),
        rag_swig_image_index_dir=_env_str("MM_EVENT_RAG_SWIG_IMAGE_INDEX_DIR", ""),
        rag_bridge_index_dir=_env_str("MM_EVENT_RAG_BRIDGE_INDEX_DIR", ""),
        rag_default_top_k=max(1, _env_int("MM_EVENT_RAG_DEFAULT_TOP_K", 3)),
        rag_text_top_k=max(1, _env_int("MM_EVENT_RAG_TEXT_TOP_K", 3)),
        rag_image_top_k=max(1, _env_int("MM_EVENT_RAG_IMAGE_TOP_K", 3)),
        rag_bridge_top_k=max(1, _env_int("MM_EVENT_RAG_BRIDGE_TOP_K", 3)),
        rag_enable_image_query=_env_flag("MM_EVENT_RAG_ENABLE_IMAGE_QUERY", True),
    )


settings = load_settings()
