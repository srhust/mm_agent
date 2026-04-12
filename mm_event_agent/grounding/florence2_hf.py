"""Florence-2 grounding executor via Hugging Face transformers.

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
import os
from copy import deepcopy
from typing import Any

from mm_event_agent.runtime_config import settings
from mm_event_agent.schemas import GroundingRequest, GroundingResult


DEFAULT_FLORENCE2_MODEL_ID = settings.florence2_model_id
DEFAULT_FLORENCE2_TASK = settings.florence2_task


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
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True)
        resolved_device = self.device
        if resolved_device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = resolved_device
        self._model.to(self.device)
        self._model.eval()

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

    def _run_single_request(self, image: Any, request: GroundingRequest) -> GroundingResult:
        """Run one Florence-2 query and normalize the output structure."""
        processor = self._processor
        model = self._model
        torch = self._torch
        if processor is None or model is None or torch is None:
            raise RuntimeError("Florence-2 model is not loaded")

        prompt = self._build_florence_query(request)
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generated_ids = model.generate(
            input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
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
        bbox, score = self._extract_best_grounding(parsed)
        if bbox is None:
            return _failed_grounding_result(request)
        return {
            "role": str(request.get("role") or ""),
            "label": str(request.get("label") or ""),
            "grounding_query": str(request.get("grounding_query") or ""),
            "bbox": bbox,
            "score": score,
            "grounding_status": "grounded",
        }

    def _extract_best_grounding(self, parsed: Any) -> tuple[list[float] | None, float | None]:
        """Extract the first usable bbox/score pair from Florence output."""
        if not isinstance(parsed, dict):
            return None, None

        task_payload = parsed.get(self.task_prompt)
        if isinstance(task_payload, dict):
            bboxes = task_payload.get("bboxes") or task_payload.get("boxes") or []
            scores = task_payload.get("scores") or []
            if isinstance(bboxes, list) and bboxes:
                bbox = _normalize_bbox(bboxes[0])
                score = _normalize_score(scores[0]) if isinstance(scores, list) and scores else None
                return bbox, score

        for value in parsed.values():
            if isinstance(value, dict):
                bboxes = value.get("bboxes") or value.get("boxes") or []
                scores = value.get("scores") or []
                if isinstance(bboxes, list) and bboxes:
                    bbox = _normalize_bbox(bboxes[0])
                    score = _normalize_score(scores[0]) if isinstance(scores, list) and scores else None
                    return bbox, score
        return None, None


def execute_grounding_requests(
    raw_image: Any,
    grounding_requests: list[GroundingRequest],
    *,
    model_id: str | None = None,
) -> list[GroundingResult]:
    """Convenience wrapper for one-off Florence-2 grounding execution."""
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
