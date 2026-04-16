"""Data node: derive image-side perception from raw_image when enabled."""

from __future__ import annotations

import base64
import hashlib
import io
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Mapping

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from mm_event_agent.observability import log_node_event
from mm_event_agent.runtime_config import settings
from mm_event_agent.schemas import extract_json_object

_PERCEPTION_CLIENT_CACHE: dict[tuple[Any, ...], ChatOpenAI] = {}
_PERCEPTION_RESULT_CACHE: dict[tuple[Any, ...], dict[str, str]] = {}


def _build_perception_summary(text: str, image_desc: str) -> str:
    normalized_text = str(text or "").strip()
    normalized_image_desc = str(image_desc or "").strip()
    return "\n".join(
        [
        f"Text: {normalized_text}" if normalized_text else "Text: ",
        f"Image: {normalized_image_desc}" if normalized_image_desc else "Image: ",
        ]
    )


def _client_cache_key() -> tuple[Any, ...]:
    return (
        str(settings.perception_backend),
        str(settings.perception_model_name),
        str(settings.perception_api_base),
        str(settings.perception_api_key),
        float(settings.perception_timeout_seconds),
    )


def _get_perception_client() -> ChatOpenAI:
    key = _client_cache_key()
    client = _PERCEPTION_CLIENT_CACHE.get(key)
    if client is None:
        kwargs: dict[str, Any] = {
            "model": settings.perception_model_name,
            "temperature": 0,
            "timeout": settings.perception_timeout_seconds,
        }
        if settings.perception_api_key:
            kwargs["api_key"] = settings.perception_api_key
        if settings.perception_api_base:
            kwargs["base_url"] = settings.perception_api_base
        client = ChatOpenAI(**kwargs)
        _PERCEPTION_CLIENT_CACHE[key] = client
    return client


def _load_image_bytes(raw_image: Any) -> tuple[bytes, str] | None:
    if raw_image is None:
        return None
    if isinstance(raw_image, (bytes, bytearray)):
        return bytes(raw_image), "image/png"
    if isinstance(raw_image, str):
        candidate = raw_image.strip()
        if not candidate:
            return None
        if candidate.startswith("data:"):
            header, _, payload = candidate.partition(",")
            if not payload:
                return None
            mime_type = header.split(";", 1)[0].replace("data:", "") or "image/png"
            try:
                return base64.b64decode(payload), mime_type
            except Exception:
                return None
        if candidate.startswith("http://") or candidate.startswith("https://"):
            return candidate.encode("utf-8"), "url"
        if not os.path.exists(candidate):
            return None
        mime_type = mimetypes.guess_type(candidate)[0] or "image/png"
        return Path(candidate).read_bytes(), mime_type
    try:
        from PIL import Image
    except Exception:
        Image = None  # type: ignore[assignment]
    if Image is not None and isinstance(raw_image, Image.Image):
        buffer = io.BytesIO()
        raw_image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue(), "image/png"
    return None


def _build_image_content_block(raw_image: Any) -> dict[str, Any] | None:
    if isinstance(raw_image, str):
        candidate = raw_image.strip()
        if candidate.startswith("http://") or candidate.startswith("https://") or candidate.startswith("data:"):
            return {"type": "image_url", "image_url": {"url": candidate}}
    payload = _load_image_bytes(raw_image)
    if payload is None:
        return None
    image_bytes, mime_type = payload
    if mime_type == "url":
        return {"type": "image_url", "image_url": {"url": image_bytes.decode("utf-8")}}
    data_url = "data:" + mime_type + ";base64," + base64.b64encode(image_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": data_url}}


def _raw_image_identity(raw_image: Any) -> str:
    if raw_image is None:
        return "none"
    if isinstance(raw_image, str):
        candidate = raw_image.strip()
        if not candidate:
            return "empty-str"
        if os.path.exists(candidate):
            resolved = Path(candidate).resolve()
            stat = resolved.stat()
            return f"path:{resolved}:{stat.st_mtime_ns}:{stat.st_size}"
        return "str:" + hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    if isinstance(raw_image, (bytes, bytearray)):
        return "bytes:" + hashlib.sha256(bytes(raw_image)).hexdigest()
    try:
        from PIL import Image
    except Exception:
        Image = None  # type: ignore[assignment]
    if Image is not None and isinstance(raw_image, Image.Image):
        buffer = io.BytesIO()
        raw_image.convert("RGB").save(buffer, format="PNG")
        return "pil:" + hashlib.sha256(buffer.getvalue()).hexdigest()
    return "repr:" + hashlib.sha256(repr(raw_image).encode("utf-8")).hexdigest()


def _perception_cache_key(raw_image: Any, *, text: str, fallback_image_desc: str) -> tuple[Any, ...]:
    return (
        str(settings.perception_backend),
        str(settings.perception_model_name),
        str(settings.perception_instruction),
        _raw_image_identity(raw_image),
        str(text or "").strip(),
        str(fallback_image_desc or "").strip(),
    )


def _msg_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
    return str(content)


def _normalize_perception_payload(raw: Any, *, text: str, fallback_image_desc: str) -> dict[str, str]:
    parsed = extract_json_object(_msg_text(raw))
    image_desc = ""
    perception_summary = ""
    if isinstance(parsed, dict):
        image_desc = str(parsed.get("image_desc") or "").strip()
        perception_summary = str(parsed.get("perception_summary") or "").strip()
    if not image_desc:
        image_desc = str(fallback_image_desc or "").strip()
    if not perception_summary:
        perception_summary = _build_perception_summary(text, image_desc)
    return {
        "image_desc": image_desc,
        "perception_summary": perception_summary,
    }


def _invoke_remote_perception(raw_image: Any, *, text: str, fallback_image_desc: str) -> dict[str, str]:
    image_block = _build_image_content_block(raw_image)
    if image_block is None:
        return {
            "image_desc": str(fallback_image_desc or "").strip(),
            "perception_summary": _build_perception_summary(text, fallback_image_desc),
        }
    prompt = str(settings.perception_instruction or "").strip()
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if str(text or "").strip():
        content.append({"type": "text", "text": f"Text context: {str(text).strip()}"})
    content.append(image_block)
    response = _get_perception_client().invoke([HumanMessage(content=content)])
    return _normalize_perception_payload(response.content, text=text, fallback_image_desc=fallback_image_desc)


def _describe_raw_image(raw_image: Any, *, text: str = "", fallback_image_desc: str = "") -> dict[str, str]:
    if not settings.use_vlm_perception:
        return {
            "image_desc": "",
            "perception_summary": _build_perception_summary(text, ""),
        }
    cache_key = _perception_cache_key(raw_image, text=text, fallback_image_desc=fallback_image_desc)
    if settings.perception_cache_enabled and cache_key in _PERCEPTION_RESULT_CACHE:
        return dict(_PERCEPTION_RESULT_CACHE[cache_key])
    result = _invoke_remote_perception(raw_image, text=text, fallback_image_desc=fallback_image_desc)
    if settings.perception_cache_enabled:
        _PERCEPTION_RESULT_CACHE[cache_key] = dict(result)
    return result


def perception(state: Mapping[str, Any]) -> dict[str, Any]:
    """Read raw text plus raw_image and write image_desc + perception_summary."""
    started_at = time.perf_counter()
    try:
        text = "" if state.get("text") is None else str(state.get("text"))
        raw_image = state.get("raw_image")
        fallback_image_desc = "" if state.get("image_desc") is None else str(state.get("image_desc")).strip()

        image_desc = fallback_image_desc
        perception_summary = _build_perception_summary(text, fallback_image_desc)
        if raw_image is not None:
            try:
                described = _describe_raw_image(raw_image, text=text, fallback_image_desc=fallback_image_desc)
            except Exception:
                described = {}
            described_image_desc = str(described.get("image_desc") or "").strip()
            if described_image_desc:
                image_desc = described_image_desc
                perception_summary = str(
                    described.get("perception_summary") or _build_perception_summary(text, image_desc)
                ).strip()
            elif not fallback_image_desc:
                perception_summary = _build_perception_summary(text, "")

        result = {"image_desc": image_desc, "perception_summary": perception_summary}
        log_node_event("perception", state, started_at, True)
        return result
    except Exception as exc:
        log_node_event("perception", state, started_at, False, error=str(exc))
        return {"image_desc": "", "perception_summary": ""}


run = perception
