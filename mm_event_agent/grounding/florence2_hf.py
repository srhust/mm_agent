"""Florence-2 grounding executor with preferred local-service integration.

raw_image is the primary image input carried in state and is the source image
used here for spatial grounding.
image_desc remains the current intermediate representation used by extraction
and verification; this module is an optional next-stage grounding layer.

This executor accepts one raw image plus multiple GroundingRequest objects and
returns structured GroundingResult objects. It does not mutate the event graph
or force bbox write-back yet.
"""

from __future__ import annotations

import io
import json
import logging
import os
from copy import deepcopy
import re
from typing import Any
from urllib import request as urllib_request

from mm_event_agent.runtime_config import settings
from mm_event_agent.schemas import GroundingRequest, GroundingResult


DEFAULT_FLORENCE2_MODEL_ID = settings.florence2_model_id
DEFAULT_FLORENCE2_TASK = settings.florence2_task
logger = logging.getLogger("mm_event_agent")


def _failed_grounding_result(request: GroundingRequest, status: str = "failed") -> GroundingResult:
    """Build a safe failed grounding result without raising."""
    return {
        "role": str(request.get("role") or ""),
        "label": str(request.get("label") or ""),
        "grounding_query": str(request.get("grounding_query") or ""),
        "bbox": None,
        "score": None,
        "grounding_status": status,
    }


def _normalize_bbox(raw_bbox: Any) -> list[float] | None:
    """Normalize Florence-style bbox outputs into [x1, y1, x2, y2]."""
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        return [float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])]
    except (TypeError, ValueError):
        return None


def _normalize_score(raw_score: Any) -> float | None:
    if raw_score is None:
        return None
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return None


def _normalize_grounding_status(raw_status: Any, bbox: list[float] | None) -> str:
    status = str(raw_status or "").strip().lower()
    if status in {"grounded", "failed", "unresolved"}:
        return status
    return "grounded" if bbox is not None else "failed"


def _normalize_service_image_ref(raw_image: Any) -> str | None:
    """Normalize raw_image into a service-safe string reference."""
    if not isinstance(raw_image, str):
        return None
    candidate = raw_image.strip()
    if not candidate:
        return None
    if candidate.startswith(("http://", "https://", "data:")):
        return candidate
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return candidate


def build_grounding_service_payload(
    raw_image: Any,
    grounding_requests: list[GroundingRequest],
    *,
    task: str,
) -> dict[str, Any] | None:
    image_ref = _normalize_service_image_ref(raw_image)
    if image_ref is None:
        return None
    normalized_requests = [
        {
            "role": str(request.get("role") or ""),
            "label": str(request.get("label") or ""),
            "grounding_query": str(request.get("grounding_query") or ""),
        }
        for request in grounding_requests
    ]
    return {
        "raw_image": image_ref,
        "grounding_requests": normalized_requests,
        "task": str(task or DEFAULT_FLORENCE2_TASK),
    }


def _normalize_service_result(
    request: GroundingRequest,
    raw_result: Any,
) -> GroundingResult:
    if not isinstance(raw_result, dict):
        return _failed_grounding_result(request)

    bbox = _normalize_bbox(raw_result.get("bbox"))
    score = _normalize_score(raw_result.get("score"))
    return {
        "role": str(raw_result.get("role") or request.get("role") or ""),
        "label": str(raw_result.get("label") or request.get("label") or ""),
        "grounding_query": str(raw_result.get("grounding_query") or request.get("grounding_query") or ""),
        "bbox": bbox,
        "score": score,
        "grounding_status": _normalize_grounding_status(raw_result.get("grounding_status"), bbox),
    }


def parse_grounding_service_response(
    raw_response: Any,
    grounding_requests: list[GroundingRequest],
) -> list[GroundingResult]:
    if not isinstance(raw_response, dict):
        return [_failed_grounding_result(request) for request in grounding_requests]

    raw_results = raw_response.get("results")
    if not isinstance(raw_results, list):
        return [_failed_grounding_result(request) for request in grounding_requests]

    normalized: list[GroundingResult] = []
    for index, request in enumerate(grounding_requests):
        raw_result = raw_results[index] if index < len(raw_results) else None
        normalized.append(_normalize_service_result(request, raw_result))
    return normalized


class Florence2ServiceGrounder:
    """HTTP client for a separate local Florence-2 grounding service."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        task_prompt: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.endpoint = str(endpoint or settings.florence2_local_endpoint or "").strip()
        self.task_prompt = task_prompt or settings.florence2_task
        self.timeout_seconds = float(timeout_seconds or settings.florence2_local_timeout_seconds)

    def execute(
        self,
        raw_image: Any,
        grounding_requests: list[GroundingRequest],
    ) -> list[GroundingResult]:
        if not grounding_requests:
            return []

        payload = build_grounding_service_payload(
            raw_image,
            grounding_requests,
            task=self.task_prompt,
        )
        if payload is None:
            logger.warning(
                "Florence grounding service payload could not be built for raw_image=%r",
                type(raw_image).__name__,
            )
            return [_failed_grounding_result(request) for request in grounding_requests]

        try:
            raw_response = self._post_json(payload)
        except Exception as exc:
            logger.warning(
                "Florence grounding service request failed endpoint=%s error=%s",
                self.endpoint,
                exc,
            )
            return [_failed_grounding_result(request) for request in grounding_requests]

        return parse_grounding_service_response(raw_response, grounding_requests)

    def _post_json(self, payload: dict[str, Any]) -> Any:
        request = urllib_request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)


class Florence2HFGrounder:
    """Minimal Florence-2 executor for grounding detector-ready queries."""

    def __init__(
        self,
        model_id: str | None = None,
        task_prompt: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id or settings.florence2_model_id
        self.task_prompt = task_prompt or settings.florence2_task
        self.device = device or settings.florence2_device or None
        self._processor = None
        self._model = None
        self._torch = None

    def execute(
        self,
        raw_image: Any,
        grounding_requests: list[GroundingRequest],
    ) -> list[GroundingResult]:
        """Run Florence-2 grounding for one raw_image and many requests.

        Fallback behavior is intentionally soft:
        - empty requests -> []
        - missing/unreadable raw_image -> failed results
        - import/model/inference errors -> failed results
        """
        if not grounding_requests:
            return []

        image = self._load_image(raw_image)
        if image is None:
            return [_failed_grounding_result(request) for request in grounding_requests]

        try:
            self._ensure_model_loaded()
        except Exception:
            return [_failed_grounding_result(request) for request in grounding_requests]

        results: list[GroundingResult] = []
        for request in grounding_requests:
            try:
                results.append(self._run_single_request(image, request))
            except Exception:
                results.append(_failed_grounding_result(request))
        return results

    def _ensure_model_loaded(self) -> None:
        if self._processor is not None and self._model is not None and self._torch is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        resolved_device = self.device
        if resolved_device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = resolved_device
        self._model.to(self.device)
        self._model.eval()

    def _is_gpu_device(self) -> bool:
        return isinstance(self.device, str) and self.device.lower().startswith("cuda")

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Move Florence inputs to the configured device with validated dtypes."""
        torch = self._torch
        if torch is None:
            raise RuntimeError("Florence-2 torch dependency is not loaded")

        prepared: dict[str, Any] = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                prepared[key] = value
                continue

            moved = value.to(self.device)
            if key == "pixel_values" and self._is_gpu_device():
                moved = moved.to(dtype=torch.float16)
            prepared[key] = moved
        return prepared

    def _load_image(self, raw_image: Any) -> Any | None:
        """Load raw_image into a PIL image if possible."""
        if raw_image is None:
            return None

        try:
            from PIL import Image
        except Exception:
            return None

        if isinstance(raw_image, Image.Image):
            return raw_image.convert("RGB")

        if isinstance(raw_image, (bytes, bytearray)):
            try:
                return Image.open(io.BytesIO(raw_image)).convert("RGB")
            except Exception:
                return None

        if isinstance(raw_image, str):
            candidate_path = raw_image.strip()
            if not candidate_path or not os.path.exists(candidate_path):
                return None
            try:
                return Image.open(candidate_path).convert("RGB")
            except Exception:
                return None

        return None

    def _build_florence_query(self, request: GroundingRequest) -> str:
        """Convert a GroundingRequest into a Florence-2 open-vocabulary query."""
        query = str(request.get("grounding_query") or "").strip()
        return f"{self.task_prompt}{query}"

    def _build_candidate_queries(self, request: GroundingRequest) -> list[str]:
        """Build deterministic phrase-grounding candidates.

        The wrapper prefers the image-side label as the primary phrase query.
        Optional fallback phrases can be carried in grounding_query using either:
        - a plain phrase distinct from the label
        - multiple phrases separated by `||` or `|`
        - a legacy `Role: label` format, whose suffix is treated as a phrase
        At most three unique candidates are returned: primary label plus up to
        two fallback phrases.
        """
        label = str(request.get("label") or "").strip()
        raw_query = str(request.get("grounding_query") or "").strip()

        candidates: list[str] = []
        seen: set[str] = set()

        def add_candidate(candidate: str) -> None:
            normalized = " ".join(candidate.split())
            if not normalized:
                return
            dedupe_key = normalized.casefold()
            if dedupe_key in seen:
                return
            seen.add(dedupe_key)
            candidates.append(normalized)

        add_candidate(label)

        if raw_query:
            pieces = [piece.strip() for piece in re.split(r"\|\|?", raw_query) if piece.strip()]
            if not pieces:
                pieces = [raw_query]
            for piece in pieces:
                add_candidate(piece)
                if ":" in piece:
                    _, suffix = piece.split(":", 1)
                    add_candidate(suffix.strip())

        return candidates[:3]

    def _run_phrase_query(self, image: Any, query: str) -> tuple[list[float] | None, float | None]:
        processor = self._processor
        model = self._model
        if processor is None or model is None:
            raise RuntimeError("Florence-2 model is not loaded")

        prompt = f"{self.task_prompt}{query}"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = self._prepare_inputs(inputs)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=self.task_prompt,
            image_size=(image.width, image.height),
        )
        return self._extract_best_grounding(parsed)

    def _run_single_request(self, image: Any, request: GroundingRequest) -> GroundingResult:
        """Run one Florence-2 query and normalize the output structure."""
        processor = self._processor
        model = self._model
        torch = self._torch
        if processor is None or model is None or torch is None:
            raise RuntimeError("Florence-2 model is not loaded")

        for query in self._build_candidate_queries(request):
            bbox, score = self._run_phrase_query(image, query)
            if bbox is not None:
                return {
                    "role": str(request.get("role") or ""),
                    "label": str(request.get("label") or ""),
                    "grounding_query": str(request.get("grounding_query") or ""),
                    "bbox": bbox,
                    "score": score,
                    "grounding_status": "grounded",
                }
        return _failed_grounding_result(request)

    def _extract_best_grounding(self, parsed: Any) -> tuple[list[float] | None, float | None]:
        """Extract the first usable bbox/score pair from Florence output."""
        if not isinstance(parsed, dict):
            return None, None

        task_payload = parsed.get(self.task_prompt)
        bbox, score = self._extract_bbox_from_payload(task_payload)
        if bbox is not None:
            return bbox, score

        for value in parsed.values():
            bbox, score = self._extract_bbox_from_payload(value)
            if bbox is not None:
                return bbox, score
        return None, None

    def _extract_bbox_from_payload(self, payload: Any) -> tuple[list[float] | None, float | None]:
        if not isinstance(payload, dict):
            return None, None

        bboxes = payload.get("bboxes") or payload.get("boxes") or []
        scores = payload.get("scores") or []
        if not isinstance(bboxes, list):
            return None, None

        for index, candidate in enumerate(bboxes):
            bbox = _normalize_bbox(candidate)
            if bbox is None:
                continue
            score = None
            if isinstance(scores, list) and index < len(scores):
                score = _normalize_score(scores[index])
            return bbox, score
        return None, None


def execute_grounding_requests(
    raw_image: Any,
    grounding_requests: list[GroundingRequest],
    *,
    model_id: str | None = None,
) -> list[GroundingResult]:
    """Convenience wrapper for one-off Florence-2 grounding execution.

    Preferred path:
    - local Florence-2 service via FLORENCE2_LOCAL_ENDPOINT

    Secondary fallback:
    - in-process HF Florence loading when no local endpoint is configured
    """
    if str(settings.florence2_local_endpoint or "").strip():
        return Florence2ServiceGrounder().execute(raw_image, grounding_requests)

    grounder = Florence2HFGrounder(model_id=model_id)
    return grounder.execute(raw_image, grounding_requests)


def apply_grounding_results_to_event(event: dict[str, Any], grounding_results: list[GroundingResult]) -> dict[str, Any]:
    """Future-facing helper to write grounded bboxes back into image_arguments.

    This helper is intentionally optional and is not wired into the graph yet.
    It only updates unresolved image arguments when a matching grounded result
    with a bbox is available.
    """
    updated_event = deepcopy(event)
    image_arguments = updated_event.get("image_arguments")
    if not isinstance(image_arguments, list):
        return updated_event

    for item in image_arguments:
        if not isinstance(item, dict):
            continue
        if item.get("bbox") is not None or item.get("grounding_status") != "unresolved":
            continue
        for result in grounding_results:
            if (
                isinstance(result, dict)
                and result.get("grounding_status") == "grounded"
                and result.get("role") == item.get("role")
                and result.get("label") == item.get("label")
                and isinstance(result.get("bbox"), list)
            ):
                item["bbox"] = list(result["bbox"])
                item["grounding_status"] = "grounded"
                break
    return updated_event
