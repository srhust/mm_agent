"""Typed data contracts and validation helpers."""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from mm_event_agent.ontology import (
    get_allowed_image_roles,
    get_allowed_text_roles,
    is_supported_event_type,
)


class ValidationError(ValueError):
    """Raised when a payload does not match the expected schema."""


class TextSpan(TypedDict):
    start: int
    end: int


class Trigger(TypedDict):
    text: str
    modality: str
    span: TextSpan | None


class TextArgument(TypedDict):
    role: str
    text: str
    span: TextSpan | None


class ImageArgument(TypedDict):
    role: str
    label: str
    bbox: list[float] | None
    grounding_status: str


class GroundingRequest(TypedDict):
    role: str
    label: str
    grounding_query: str
    grounding_status: str


class GroundingResult(TypedDict):
    role: str
    label: str
    grounding_query: str
    bbox: list[float] | None
    score: float | None
    grounding_status: str


class GroundingSummary(TypedDict):
    unresolved_image_arguments: int
    grounding_requests: int
    grounded_results: int
    failed_grounding_results: int
    applied_grounded_bboxes: int


class EvidenceSourceSummary(TypedDict):
    text_support: bool
    image_support: bool
    grounding_support: bool
    external_evidence_support: bool


class EvidenceSourceSnapshot(TypedDict):
    event_type: str
    text_support: bool
    image_support: bool
    grounding_support: bool
    external_evidence_support: bool
    final_event: Event


class Event(TypedDict):
    event_type: str
    trigger: Trigger | None
    text_arguments: list[TextArgument]
    image_arguments: list[ImageArgument]


class EvidenceItem(TypedDict):
    title: str
    snippet: str
    url: str
    source_type: str
    published_at: str | None
    score: float


class TextEventExample(TypedDict):
    id: str
    event_type: str
    modality: str
    raw_text: str
    trigger: Trigger | dict[str, Any]
    text_arguments: list[TextArgument | dict[str, Any]]
    source_dataset: str
    pattern_summary: str
    retrieval_text: str


class ImageSemanticExample(TypedDict):
    id: str
    source_dataset: str
    modality: str
    event_type: str
    visual_situation: str
    image_desc: str
    image_arguments: list[dict[str, Any]]
    visual_pattern_summary: str
    retrieval_text: str


class BridgeExample(TypedDict):
    id: str
    source_dataset: str
    modality: str
    event_type: str
    role: str
    text_cues: list[str]
    visual_cues: list[str]
    note: str
    retrieval_text: str


class LayeredSimilarEvents(TypedDict):
    text_event_examples: list[TextEventExample]
    image_semantic_examples: list[ImageSemanticExample]
    bridge_examples: list[BridgeExample]


class RagRetrievalMetadata(TypedDict):
    score: float
    rank: int
    index_name: str


class RagDocumentMeta(TypedDict, total=False):
    doc_id: str
    image_id: str
    path: str
    source_dataset: str
    modality: str
    event_type: str
    index_name: str
    shard_id: str
    row: int
    source_path: str
    retrieval_text: str


class RagIndexBuildInfo(TypedDict, total=False):
    index_name: str
    encoder_type: str
    index_type: str
    encoder_name: str
    encoder_name_or_path: str
    instruction: str
    batch_size: int
    vector_dim: int
    normalized: bool
    doc_count: int
    record_count: int
    built_at: str
    index_path: str
    meta_path: str
    source_path: str


class RagQuery(TypedDict, total=False):
    raw_text: str
    image_desc: str
    event_type: str
    top_k: int
    text_top_k: int
    image_top_k: int
    bridge_top_k: int
    enable_image_query: bool


class VerificationDiagnostic(TypedDict):
    field_path: str
    issue_type: str
    suggested_action: str


class FusionContext(TypedDict):
    raw_text: str
    # Current image-side intermediate representation derived from raw_image,
    # not the original raw image object itself.
    raw_image_desc: str
    perception_summary: str
    text_tokens: list[str]
    patterns: LayeredSimilarEvents
    evidence: list[EvidenceItem]


def empty_event() -> Event:
    return {
        "event_type": "",
        "trigger": None,
        "text_arguments": [],
        "image_arguments": [],
    }


def empty_fusion_context() -> FusionContext:
    """Create the current fusion context.

    raw_text and raw_image are the primary user inputs at graph entry.
    fusion_context.raw_image_desc stores the derived image description used by
    the current downstream extraction / verification path.
    The original raw_image continues to travel separately in the global state.
    """
    return {
        "raw_text": "",
        "raw_image_desc": "",
        "perception_summary": "",
        "text_tokens": [],
        "patterns": empty_layered_similar_events(),
        "evidence": [],
    }


def empty_layered_similar_events() -> LayeredSimilarEvents:
    return {
        "text_event_examples": [],
        "image_semantic_examples": [],
        "bridge_examples": [],
    }


def attach_retrieval_metadata(
    item: dict[str, Any],
    *,
    score: float,
    rank: int,
    index_name: str,
) -> dict[str, Any]:
    """Attach retrieval metadata without mutating the caller's input object."""
    enriched = dict(item)
    enriched["retrieval_metadata"] = {
        "score": float(score),
        "rank": int(rank),
        "index_name": str(index_name or "").strip(),
    }
    return enriched


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", re.UNICODE)
_DETERMINER_TOKENS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
}
_QUANTITY_TOKENS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "several",
    "many",
    "few",
    "multiple",
    "single",
    "dozens",
    "dozen",
    "hundreds",
    "thousands",
}
_PERSON_TITLE_MODIFIER_TOKENS = {
    "former",
}
_PERSON_TITLE_TOKENS = {
    "president",
    "mr",
    "mrs",
    "ms",
    "dr",
    "officer",
    "mayor",
    "governor",
    "senator",
    "sen",
    "representative",
    "rep",
    "judge",
    "justice",
    "professor",
    "general",
    "coach",
    "captain",
}
_PERSON_TITLE_COMPOUND_PREFIXES = {
    ("police", "officer"),
}
_PERSON_NAME_PARTICLE_TOKENS = {
    "al",
    "bin",
    "da",
    "de",
    "del",
    "der",
    "di",
    "dos",
    "du",
    "la",
    "le",
    "st",
    "van",
    "von",
}


def tokenize_text(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(str(text or ""))


def resolve_text_token_sequence(raw_text: str, token_sequence: Any = None) -> list[str]:
    if isinstance(token_sequence, list):
        normalized = [str(item) for item in token_sequence if str(item)]
        if normalized:
            return normalized
    return tokenize_text(raw_text)


def extract_json_object(text: str) -> dict[str, Any] | None:
    payload = text.strip()
    payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        data = json.loads(payload)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    start, end = payload.find("{"), payload.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(payload[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def validate_event(data: Any) -> Event:
    if not isinstance(data, dict):
        raise ValidationError("event must be a JSON object")

    event_type = data.get("event_type")
    trigger = data.get("trigger")
    text_arguments = data.get("text_arguments")
    image_arguments = data.get("image_arguments")

    if not isinstance(event_type, str):
        raise ValidationError("event.event_type must be a string")
    if not is_supported_event_type(event_type.strip()):
        raise ValidationError("event.event_type must be in the supported ontology")
    if trigger is not None and not isinstance(trigger, dict):
        raise ValidationError("event.trigger must be an object or null")
    if not isinstance(text_arguments, list):
        raise ValidationError("event.text_arguments must be a list")
    if not isinstance(image_arguments, list):
        raise ValidationError("event.image_arguments must be a list")

    normalized_event_type = event_type.strip()
    normalized_trigger = _validate_trigger(trigger)
    normalized_text_arguments = [
        _validate_text_argument(item, normalized_event_type) for item in text_arguments
    ]
    normalized_image_arguments = [
        _validate_image_argument(item, normalized_event_type) for item in image_arguments
    ]

    return {
        "event_type": normalized_event_type,
        "trigger": normalized_trigger,
        "text_arguments": normalized_text_arguments,
        "image_arguments": normalized_image_arguments,
    }


def parse_event_json(text: str) -> Event:
    data = extract_json_object(text)
    if data is None:
        raise ValidationError("event output is not valid JSON")
    return validate_event(data)


def _is_matching_span(span: TextSpan | None, source_tokens: list[str], value: str) -> bool:
    normalized_span = normalize_text_span(span)
    if normalized_span is None:
        return False
    start = normalized_span["start"]
    end = normalized_span["end"]
    if start < 0 or end < start or end > len(source_tokens):
        return False
    return source_tokens[start:end] == tokenize_text(value)


def _unique_exact_span(source_tokens: list[str], value: str) -> TextSpan | None:
    occurrences = find_all_text_occurrences(source_tokens, value)
    if len(occurrences) == 1:
        return occurrences[0]
    return None


def _is_punctuation_token(token: str) -> bool:
    return bool(token) and bool(re.fullmatch(r"[^\w\s]+", token))


def _is_quantity_token(token: str) -> bool:
    normalized = str(token or "").strip().lower()
    if not normalized:
        return False
    if normalized in _QUANTITY_TOKENS:
        return True
    return normalized.isdigit()


def _looks_like_proper_name(tokens: list[str]) -> bool:
    lexical = [token for token in tokens if token and not _is_punctuation_token(token)]
    if len(lexical) < 2:
        return False
    for token in lexical:
        if not token[0].isupper():
            return False
    return True


def _normalize_person_prefix_token(token: str) -> str:
    return re.sub(r"[^a-z]+", "", str(token or "").strip().lower())


def _looks_like_person_name(tokens: list[str]) -> bool:
    lexical = [str(token or "").strip() for token in tokens if str(token or "").strip() and not _is_punctuation_token(str(token))]
    if not lexical:
        return False
    for index, token in enumerate(lexical):
        normalized = _normalize_person_prefix_token(token)
        if index > 0 and normalized in _PERSON_NAME_PARTICLE_TOKENS:
            continue
        if not token or not token[0].isupper():
            return False
    return True


def _preferred_person_name_token_slice(tokens: list[str]) -> tuple[int, int] | None:
    if not tokens:
        return None
    normalized = [_normalize_person_prefix_token(token) for token in tokens]
    best: tuple[int, int] | None = None
    for prefix_len in range(1, len(tokens)):
        suffix = tokens[prefix_len:]
        if not _looks_like_person_name(suffix):
            continue
        prefix = normalized[:prefix_len]
        has_title = any(token in _PERSON_TITLE_TOKENS for token in prefix)
        if tuple(prefix) in _PERSON_TITLE_COMPOUND_PREFIXES:
            has_title = True
        if prefix_len >= 2 and tuple(prefix[-2:]) in _PERSON_TITLE_COMPOUND_PREFIXES:
            has_title = True
        if not has_title:
            continue
        prefix_is_allowed = True
        for index, token in enumerate(prefix):
            if token in _PERSON_TITLE_TOKENS or token in _PERSON_TITLE_MODIFIER_TOKENS or not token:
                continue
            pair_with_next = index + 1 < len(prefix) and (token, prefix[index + 1]) in _PERSON_TITLE_COMPOUND_PREFIXES
            pair_with_prev = index > 0 and (prefix[index - 1], token) in _PERSON_TITLE_COMPOUND_PREFIXES
            if not pair_with_next and not pair_with_prev:
                prefix_is_allowed = False
                break
        if prefix_is_allowed:
            best = (prefix_len, len(tokens))
    return best


def _preferred_argument_token_slice(tokens: list[str]) -> tuple[int, int]:
    if not tokens:
        return 0, 0
    start = 0
    end = len(tokens)

    while start < end and _is_punctuation_token(tokens[start]):
        start += 1
    while end > start and _is_punctuation_token(tokens[end - 1]):
        end -= 1
    while start < end and (
        tokens[start].lower() in _DETERMINER_TOKENS
        or _is_quantity_token(tokens[start])
    ):
        start += 1

    if start >= end:
        return 0, len(tokens)

    trimmed = tokens[start:end]
    person_slice = _preferred_person_name_token_slice(trimmed)
    if person_slice is not None:
        return start + person_slice[0], start + person_slice[1]
    if _looks_like_proper_name(trimmed):
        return start, end
    if len(trimmed) == 1:
        return start, end
    return end - 1, end


def normalize_text_argument_boundary(
    text: str,
    span: TextSpan | None,
    source_tokens: list[str],
) -> tuple[str, TextSpan | None]:
    normalized_span = normalize_text_span(span)
    if normalized_span is None:
        return str(text or "").strip(), None
    start = normalized_span["start"]
    end = normalized_span["end"]
    if start < 0 or end > len(source_tokens) or end <= start:
        return str(text or "").strip(), normalized_span
    mention_tokens = source_tokens[start:end]
    slice_start, slice_end = _preferred_argument_token_slice(mention_tokens)
    shrunk_tokens = mention_tokens[slice_start:slice_end]
    if not shrunk_tokens:
        shrunk_tokens = mention_tokens
        slice_start = 0
        slice_end = len(mention_tokens)
    return (
        " ".join(shrunk_tokens),
        {"start": start + slice_start, "end": start + slice_end},
    )


def describe_text_argument_normalization(
    text: str,
    span: TextSpan | None,
    source_tokens: list[str],
) -> dict[str, Any]:
    normalized_text, normalized_span = normalize_text_argument_boundary(text, span, source_tokens)
    original_text = str(text or "").strip()
    original_span = normalize_text_span(span)
    original_tokens = tokenize_text(original_text)
    details = {
        "normalized_text": normalized_text,
        "normalized_span": normalized_span,
        "has_determiner": False,
        "has_quantity": False,
        "has_person_title_prefix": False,
        "is_broader_than_preferred": False,
    }
    if original_span is None:
        return details
    span_tokens = source_tokens[original_span["start"] : original_span["end"]]
    if span_tokens:
        details["has_determiner"] = span_tokens[0].lower() in _DETERMINER_TOKENS
        details["has_quantity"] = _is_quantity_token(span_tokens[0])
        preferred_person_slice = _preferred_person_name_token_slice(span_tokens)
        details["has_person_title_prefix"] = preferred_person_slice is not None and preferred_person_slice[0] > 0
    details["is_broader_than_preferred"] = (
        normalized_span is not None
        and (
            normalized_span["start"] != original_span["start"]
            or normalized_span["end"] != original_span["end"]
            or tokenize_text(normalized_text) != original_tokens
        )
    )
    return details


def align_text_grounded_event(
    event: Event,
    raw_text: str,
    token_sequence: Any = None,
) -> tuple[Event, list[str], list[VerificationDiagnostic]]:
    issues: list[str] = []
    diagnostics: list[VerificationDiagnostic] = []
    source_tokens = resolve_text_token_sequence(raw_text, token_sequence)

    trigger = event["trigger"]
    aligned_trigger: Trigger | None = None
    if trigger is not None:
        trigger_text = str(trigger.get("text") or "")
        trigger_span = normalize_text_span(trigger.get("span"))
        if _is_matching_span(trigger_span, source_tokens, trigger_text):
            aligned_trigger = {
                "text": trigger_text,
                "modality": "text",
                "span": normalize_text_span(trigger_span),
            }
        else:
            realigned = _unique_exact_span(source_tokens, trigger_text)
            if realigned is not None:
                aligned_trigger = {
                    "text": trigger_text,
                    "modality": "text",
                    "span": realigned,
                }
            else:
                issue_type = "span_mismatch" if trigger_span is not None else "missing_span_alignment"
                issues.append("trigger dropped during strict text alignment")
                diagnostics.append(
                    {
                        "field_path": "trigger.span",
                        "issue_type": issue_type,
                        "suggested_action": "realign_or_drop",
                    }
                )

    aligned_text_arguments: list[TextArgument] = []
    for index, item in enumerate(event["text_arguments"]):
        text = str(item.get("text") or "")
        span = normalize_text_span(item.get("span"))
        if _is_matching_span(span, source_tokens, text):
            normalized_text, normalized_span = normalize_text_argument_boundary(text, span, source_tokens)
            aligned_text_arguments.append(
                {
                    "role": item["role"],
                    "text": normalized_text,
                    "span": normalized_span,
                }
            )
            continue
        realigned = _unique_exact_span(source_tokens, text)
        if realigned is not None:
            normalized_text, normalized_span = normalize_text_argument_boundary(text, realigned, source_tokens)
            aligned_text_arguments.append(
                {
                    "role": item["role"],
                    "text": normalized_text,
                    "span": normalized_span,
                }
            )
            continue
        issue_type = "span_mismatch" if span is not None else "missing_span_alignment"
        issues.append(f"text argument dropped during strict text alignment at index {index}")
        diagnostics.append(
            {
                "field_path": f"text_arguments[{index}].span",
                "issue_type": issue_type,
                "suggested_action": "realign_or_drop",
            }
        )

    return (
        {
            "event_type": event["event_type"],
            "trigger": aligned_trigger,
            "text_arguments": aligned_text_arguments,
            "image_arguments": event["image_arguments"],
        },
        issues,
        diagnostics,
    )


def enforce_strict_text_grounding(event: Event, raw_text: str, token_sequence: Any = None) -> Event:
    aligned, _, _ = align_text_grounded_event(event, raw_text, token_sequence=token_sequence)
    return aligned


def attach_text_spans(event: Event, raw_text: str, token_sequence: Any = None) -> Event:
    """Backward-compatible alias for strict text grounding post-processing."""
    return enforce_strict_text_grounding(event, raw_text, token_sequence=token_sequence)


def image_argument_needs_grounding(data: Any) -> bool:
    """Return True when an image argument is still unresolved and detector-ready."""
    if not isinstance(data, dict):
        return False
    role = str(data.get("role") or "").strip()
    label = str(data.get("label") or "").strip()
    bbox = data.get("bbox")
    grounding_status = str(data.get("grounding_status") or "").strip()
    return bool(role and label and bbox is None and grounding_status == "unresolved")


def build_grounding_query(role: str, label: str) -> str:
    """Build a lightweight detector-facing query from semantic role + label."""
    normalized_role = str(role or "").strip()
    normalized_label = str(label or "").strip()
    if normalized_role and normalized_label:
        return f"{normalized_role}: {normalized_label}"
    return normalized_label or normalized_role


def build_grounding_request(image_argument: Any) -> GroundingRequest | None:
    """Convert one unresolved image argument into a detector-ready request."""
    if not image_argument_needs_grounding(image_argument):
        return None
    role = str(image_argument.get("role") or "").strip()
    label = str(image_argument.get("label") or "").strip()
    return {
        "role": role,
        "label": label,
        "grounding_query": build_grounding_query(role, label),
        "grounding_status": "unresolved",
    }


def build_grounding_requests(event: Any) -> list[GroundingRequest]:
    """Derive detector-ready requests for unresolved image arguments only."""
    if not isinstance(event, dict):
        return []
    image_arguments = event.get("image_arguments")
    if not isinstance(image_arguments, list):
        return []

    requests: list[GroundingRequest] = []
    for item in image_arguments:
        request = build_grounding_request(item)
        if request is not None:
            requests.append(request)
    return requests


def validate_evidence_item(data: Any) -> EvidenceItem:
    if not isinstance(data, dict):
        raise ValidationError("evidence item must be an object")

    title = data.get("title")
    snippet = data.get("snippet")
    url = data.get("url")
    source_type = data.get("source_type")
    published_at = data.get("published_at")
    raw_score = data.get("score", 0.0)

    if not isinstance(title, str) or not title.strip():
        raise ValidationError("evidence.title must be a non-empty string")
    if not isinstance(snippet, str) or not snippet.strip():
        raise ValidationError("evidence.snippet must be a non-empty string")
    if not isinstance(url, str) or not url.strip():
        raise ValidationError("evidence.url must be a non-empty string")
    if not isinstance(source_type, str) or not source_type.strip():
        raise ValidationError("evidence.source_type must be a non-empty string")
    if published_at is not None and not isinstance(published_at, str):
        raise ValidationError("evidence.published_at must be a string or null")

    try:
        score = float(raw_score)
    except (TypeError, ValueError) as exc:
        raise ValidationError("evidence.score must be numeric") from exc

    return {
        "title": title.strip(),
        "snippet": snippet.strip(),
        "url": url.strip(),
        "source_type": source_type.strip(),
        "published_at": published_at.strip() if isinstance(published_at, str) and published_at.strip() else None,
        "score": max(0.0, min(1.0, score)),
    }


def _validate_trigger(data: Any) -> Trigger | None:
    if data is None:
        return None
    text = data.get("text")
    modality = data.get("modality")
    span = data.get("span")
    if not isinstance(text, str):
        raise ValidationError("event.trigger.text must be a string")
    if modality != "text":
        raise ValidationError('event.trigger.modality must be "text"')
    return {
        "text": text,
        "modality": "text",
        "span": _validate_span(span),
    }


def _validate_text_argument(data: Any, event_type: str) -> TextArgument:
    if not isinstance(data, dict):
        raise ValidationError("event.text_arguments items must be objects")
    role = data.get("role")
    text = data.get("text")
    span = data.get("span")
    if not isinstance(role, str) or not role.strip():
        raise ValidationError("text argument role must be a non-empty string")
    if not isinstance(text, str) or not text.strip():
        raise ValidationError("text argument text must be a non-empty string")
    normalized_role = role.strip()
    if normalized_role not in get_allowed_text_roles(event_type):
        raise ValidationError(f'text argument role "{normalized_role}" is not allowed for event.event_type')
    return {
        "role": normalized_role,
        "text": text,
        "span": _validate_span(span),
    }


def _validate_image_argument(data: Any, event_type: str) -> ImageArgument:
    if not isinstance(data, dict):
        raise ValidationError("event.image_arguments items must be objects")
    role = data.get("role")
    label = data.get("label")
    bbox = data.get("bbox")
    grounding_status = data.get("grounding_status")
    if not isinstance(role, str) or not role.strip():
        raise ValidationError("image argument role must be a non-empty string")
    if not isinstance(label, str) or not label.strip():
        raise ValidationError("image argument label must be a non-empty string")
    if not isinstance(grounding_status, str) or not grounding_status.strip():
        raise ValidationError("image argument grounding_status must be a non-empty string")
    normalized_role = role.strip()
    if normalized_role not in get_allowed_image_roles(event_type):
        raise ValidationError(f'image argument role "{normalized_role}" is not allowed for event.event_type')

    norm_bbox: list[float] | None
    if bbox is None:
        if grounding_status != "unresolved":
            raise ValidationError("image argument without bbox must be marked unresolved")
        norm_bbox = None
    else:
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValidationError("image argument bbox must be a list of four numbers or null")
        norm_bbox = []
        for value in bbox:
            try:
                norm_bbox.append(float(value))
            except (TypeError, ValueError) as exc:
                raise ValidationError("image argument bbox values must be numeric") from exc
    return {
        "role": normalized_role,
        "label": label.strip(),
        "bbox": norm_bbox,
        "grounding_status": grounding_status.strip(),
    }


def _validate_span(span: Any) -> TextSpan | None:
    normalized_span = normalize_text_span(span)
    if normalized_span is None:
        if span is None:
            return None
        raise ValidationError('span must be null, {"start": int, "end": int}, or [start, end]')
    if normalized_span["start"] < 0 or normalized_span["end"] < normalized_span["start"]:
        raise ValidationError("span values must be valid non-negative integers")
    return normalized_span


def normalize_text_span(span: Any) -> TextSpan | None:
    if span is None:
        return None
    if isinstance(span, dict):
        start = span.get("start")
        end = span.get("end")
    elif isinstance(span, list) and len(span) == 2:
        start, end = span
    else:
        return None
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    return {"start": start, "end": end}


def find_all_text_occurrences(source_text: Any, value: str) -> list[TextSpan]:
    source_tokens = (
        [str(item) for item in source_text]
        if isinstance(source_text, list)
        else resolve_text_token_sequence(str(source_text or ""))
    )
    target_tokens = tokenize_text(value)
    if not source_tokens or not target_tokens:
        return []

    matches: list[TextSpan] = []
    target_len = len(target_tokens)
    max_start = len(source_tokens) - target_len
    for start in range(max_start + 1):
        if source_tokens[start : start + target_len] == target_tokens:
            matches.append({"start": start, "end": start + target_len})
    return matches


def choose_best_span(
    occurrences: list[TextSpan],
    anchor_spans: list[TextSpan] | None = None,
) -> TextSpan | None:
    if not occurrences:
        return None
    if len(occurrences) == 1:
        return occurrences[0]

    anchors = [
        span
        for span in (anchor_spans or [])
        if isinstance(span, dict)
        and isinstance(span.get("start"), int)
        and isinstance(span.get("end"), int)
    ]
    if not anchors:
        return None

    scored: list[tuple[float, TextSpan]] = []
    for occurrence in occurrences:
        occ_mid = (occurrence["start"] + occurrence["end"]) / 2.0
        min_distance = min(
            abs(occ_mid - ((anchor["start"] + anchor["end"]) / 2.0))
            for anchor in anchors
        )
        scored.append((min_distance, occurrence))

    scored.sort(key=lambda item: item[0])
    if len(scored) >= 2 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def find_text_span(
    source_text: str,
    value: str,
    token_sequence: Any = None,
    anchor_spans: list[TextSpan] | None = None,
) -> TextSpan | None:
    occurrences = find_all_text_occurrences(resolve_text_token_sequence(source_text, token_sequence), value)
    return choose_best_span(occurrences, anchor_spans=anchor_spans)
