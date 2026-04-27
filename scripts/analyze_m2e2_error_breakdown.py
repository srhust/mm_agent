from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mm_event_agent.m2e2_adapter import get_m2e2_sample_id
from mm_event_agent.ontology import get_allowed_roles
from scripts.score_m2e2_current import (
    bbox_iou,
    extract_gold_image_arguments,
    extract_gold_text_argument_records,
    extract_predicted_image_arguments,
    extract_predicted_text_argument_records,
    load_json_or_jsonl,
    match_image_arguments,
    normalize_bbox,
    resolve_gold_event_type,
    resolve_predicted_event_type,
)


TEXT_COUNTERS = (
    "text.true_positive",
    "text.event_type_mismatch_samples",
    "text.fp_due_to_wrong_event_type",
    "text.fp_wrong_role_same_span",
    "text.fp_boundary_error_same_role",
    "text.fp_wrong_role_and_boundary",
    "text.fp_role_outside_schema",
    "text.fp_spurious_argument",
    "text.fn_due_to_missing_event_type",
    "text.fn_wrong_role_same_span",
    "text.fn_boundary_error_same_role",
    "text.fn_wrong_role_and_boundary",
    "text.fn_missing_argument",
    "text.pred_role_illegal_under_pred_event",
    "text.pred_role_outside_gold_schema_union",
)

IMAGE_COUNTERS = (
    "image.true_positive",
    "image.event_type_mismatch_samples",
    "image.fp_due_to_wrong_event_type",
    "image.fp_wrong_role_correct_box",
    "image.fp_box_not_precise_enough_oversized",
    "image.fp_box_not_precise_enough_undersized",
    "image.fp_box_not_precise_enough_shifted",
    "image.fp_weak_overlap_same_role",
    "image.fp_no_overlap_same_role",
    "image.fp_spurious_argument",
    "image.fn_due_to_missing_event_type",
    "image.fn_wrong_role_correct_box",
    "image.fn_box_not_precise_enough_oversized",
    "image.fn_box_not_precise_enough_undersized",
    "image.fn_box_not_precise_enough_shifted",
    "image.fn_weak_overlap_same_role",
    "image.fn_no_overlap_same_role",
    "image.fn_missing_argument",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce fine-grained scorer-style M2E2 text/image argument error breakdowns."
    )
    parser.add_argument("--gold", required=True, help="Path to gold M2E2 JSON or JSONL samples.")
    parser.add_argument("--pred", required=True, help="Path to predictions.jsonl.")
    parser.add_argument("--trace", help="Optional trace.jsonl for lightweight attribution hints.")
    parser.add_argument("--per-sample-metrics", help="Optional per_sample_metrics.jsonl.")
    parser.add_argument("--save-json", help="Optional path for the detailed JSON artifact.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="TP IoU threshold. Default: 0.5.")
    parser.add_argument("--iou-low", type=float, default=0.2, help="Weak-overlap lower bound. Default: 0.2.")
    parser.add_argument("--cover-high", type=float, default=0.8, help="Coverage threshold for oversized boxes.")
    parser.add_argument("--purity-low", type=float, default=0.6, help="Purity threshold for oversized boxes.")
    parser.add_argument("--sample-limit", type=int, default=5, help="Examples stored per bucket. Default: 5.")
    return parser.parse_args()


def index_records_by_id(
    records: Iterable[Mapping[str, Any]],
    *,
    id_keys: tuple[str, ...] = ("sample_id", "id"),
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        for key in id_keys:
            value = record.get(key)
            if value is not None and str(value).strip():
                indexed[str(value).strip()] = dict(record)
                break
    return indexed


def analyze_error_breakdown(
    gold_samples: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    *,
    trace_records: list[dict[str, Any]] | None = None,
    per_sample_metrics_records: list[dict[str, Any]] | None = None,
    iou_threshold: float = 0.5,
    iou_low: float = 0.2,
    cover_high: float = 0.8,
    purity_low: float = 0.6,
    sample_limit: int = 5,
) -> dict[str, Any]:
    predictions_by_id = index_records_by_id(prediction_records, id_keys=("id", "sample_id"))
    traces_by_id = index_records_by_id(trace_records or [])
    metrics_by_id = index_records_by_id(per_sample_metrics_records or [])

    counters: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    processed = 0

    for gold_sample in gold_samples:
        processed += 1
        sample_id = get_m2e2_sample_id(gold_sample) or str(gold_sample.get("id") or "")
        pred_record = predictions_by_id.get(sample_id)
        trace_record = traces_by_id.get(sample_id)
        metrics_record = metrics_by_id.get(sample_id)
        context = _sample_context(gold_sample, pred_record, trace_record, metrics_record)

        _analyze_text_sample(
            gold_sample,
            pred_record,
            counters=counters,
            examples=examples,
            context=context,
            sample_limit=sample_limit,
        )
        _analyze_image_sample(
            gold_sample,
            pred_record,
            counters=counters,
            examples=examples,
            context=context,
            iou_threshold=iou_threshold,
            iou_low=iou_low,
            cover_high=cover_high,
            purity_low=purity_low,
            sample_limit=sample_limit,
        )

    text_summary = {key: int(counters.get(key, 0)) for key in TEXT_COUNTERS}
    image_summary = {key: int(counters.get(key, 0)) for key in IMAGE_COUNTERS}
    return {
        "summary": {
            "overall": {
                "num_gold_samples": len(gold_samples),
                "num_prediction_records": len(prediction_records),
                "num_records": processed,
                "text_error_total": sum(value for key, value in text_summary.items() if key.startswith("text.fp_") or key.startswith("text.fn_")),
                "image_error_total": sum(value for key, value in image_summary.items() if key.startswith("image.fp_") or key.startswith("image.fn_")),
            },
            "text": text_summary,
            "image": image_summary,
        },
        "example_samples": {key: value for key, value in sorted(examples.items())},
        "num_records": processed,
        "iou_threshold": iou_threshold,
        "iou_low": iou_low,
        "cover_high": cover_high,
        "purity_low": purity_low,
    }


def _analyze_text_sample(
    gold_sample: Mapping[str, Any],
    pred_record: Mapping[str, Any] | None,
    *,
    counters: Counter[str],
    examples: dict[str, list[dict[str, Any]]],
    context: dict[str, Any],
    sample_limit: int,
) -> None:
    gold_event_type = context["gold_event_type"]
    pred_event_type = context["predicted_event_type"]
    gold_records = extract_gold_text_argument_records(gold_sample, ignore_trigger=False)
    pred_records = extract_predicted_text_argument_records(pred_record, ignore_trigger=False)
    unmatched_gold, unmatched_pred = _match_exact_text(gold_records, pred_records)

    text_tp = len(gold_records) - len(unmatched_gold)
    counters["text.true_positive"] += text_tp
    if gold_event_type and pred_event_type and gold_event_type != pred_event_type:
        counters["text.event_type_mismatch_samples"] += 1

    gold_schema_roles = set(get_allowed_roles(gold_event_type))
    pred_schema_roles = set(get_allowed_roles(pred_event_type))
    for pred_index in unmatched_pred:
        pred_item = pred_records[pred_index]
        if pred_event_type and pred_item["role"] not in pred_schema_roles:
            _add_counter("text.pred_role_illegal_under_pred_event", counters, examples, context, sample_limit, pred=pred_item)
        if gold_schema_roles and pred_item["role"] not in gold_schema_roles:
            _add_counter("text.pred_role_outside_gold_schema_union", counters, examples, context, sample_limit, pred=pred_item)

        if gold_event_type and pred_event_type and gold_event_type != pred_event_type:
            _add_counter("text.fp_due_to_wrong_event_type", counters, examples, context, sample_limit, pred=pred_item)
            continue
        bucket, gold_match = _classify_unmatched_text_pred(pred_item, [gold_records[i] for i in unmatched_gold], gold_schema_roles)
        _add_counter(bucket, counters, examples, context, sample_limit, pred=pred_item, gold=gold_match)

    for gold_index in unmatched_gold:
        gold_item = gold_records[gold_index]
        if gold_event_type and not pred_event_type:
            _add_counter("text.fn_due_to_missing_event_type", counters, examples, context, sample_limit, gold=gold_item)
            continue
        bucket, pred_match = _classify_unmatched_text_gold(gold_item, [pred_records[i] for i in unmatched_pred])
        _add_counter(bucket, counters, examples, context, sample_limit, gold=gold_item, pred=pred_match)


def _analyze_image_sample(
    gold_sample: Mapping[str, Any],
    pred_record: Mapping[str, Any] | None,
    *,
    counters: Counter[str],
    examples: dict[str, list[dict[str, Any]]],
    context: dict[str, Any],
    iou_threshold: float,
    iou_low: float,
    cover_high: float,
    purity_low: float,
    sample_limit: int,
) -> None:
    gold_event_type = context["gold_event_type"]
    pred_event_type = context["predicted_event_type"]
    gold_image = extract_gold_image_arguments(gold_sample)
    pred_image = extract_predicted_image_arguments(pred_record)
    invalid_pred_image = _extract_invalid_predicted_image_arguments(pred_record)

    image_tp, image_matches = match_image_arguments(pred_image, gold_image, image_iou=iou_threshold)
    counters["image.true_positive"] += image_tp
    if gold_event_type and pred_event_type and gold_event_type != pred_event_type:
        counters["image.event_type_mismatch_samples"] += 1

    matched_pred = {match["pred_index"] for match in image_matches if match.get("matched")}
    matched_gold = {match["matched_gold_index"] for match in image_matches if match.get("matched")}
    unmatched_pred = [index for index in range(len(pred_image)) if index not in matched_pred]
    unmatched_gold = [index for index in range(len(gold_image)) if index not in matched_gold]

    for pred_index in unmatched_pred:
        pred_item = pred_image[pred_index]
        if gold_event_type and pred_event_type and gold_event_type != pred_event_type:
            _add_counter("image.fp_due_to_wrong_event_type", counters, examples, context, sample_limit, pred=pred_item)
            continue
        bucket, gold_match = _classify_unmatched_image_pred(
            pred_item,
            [gold_image[i] for i in unmatched_gold],
            iou_threshold=iou_threshold,
            iou_low=iou_low,
            cover_high=cover_high,
            purity_low=purity_low,
        )
        _add_counter(bucket, counters, examples, context, sample_limit, pred=pred_item, gold=gold_match)

    for gold_index in unmatched_gold:
        gold_item = gold_image[gold_index]
        if gold_event_type and not pred_event_type:
            _add_counter("image.fn_due_to_missing_event_type", counters, examples, context, sample_limit, gold=gold_item)
            continue
        bucket, pred_match = _classify_unmatched_image_gold(
            gold_item,
            [pred_image[i] for i in unmatched_pred],
            iou_threshold=iou_threshold,
            iou_low=iou_low,
            cover_high=cover_high,
            purity_low=purity_low,
        )
        _add_counter(bucket, counters, examples, context, sample_limit, gold=gold_item, pred=pred_match)

    for invalid in invalid_pred_image:
        _add_counter("image.fp_spurious_argument", counters, examples, context, sample_limit, pred=invalid)


def _match_exact_text(
    gold_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    unmatched_gold = set(range(len(gold_records)))
    unmatched_pred: list[int] = []
    for pred_index, pred_item in enumerate(pred_records):
        match_index = None
        for gold_index in sorted(unmatched_gold):
            if gold_records[gold_index]["tuple"] == pred_item["tuple"]:
                match_index = gold_index
                break
        if match_index is None:
            unmatched_pred.append(pred_index)
        else:
            unmatched_gold.remove(match_index)
    return sorted(unmatched_gold), unmatched_pred


def _classify_unmatched_text_pred(
    pred_item: Mapping[str, Any],
    gold_candidates: list[dict[str, Any]],
    gold_schema_roles: set[str],
) -> tuple[str, Mapping[str, Any] | None]:
    same_span = [item for item in gold_candidates if _same_text_context(item, pred_item) and _same_span(item, pred_item)]
    if same_span:
        return "text.fp_wrong_role_same_span", same_span[0]

    same_role_overlap = [
        item for item in gold_candidates if _same_text_event_and_trigger(item, pred_item) and item["role"] == pred_item["role"] and _span_overlap(item, pred_item)
    ]
    if same_role_overlap:
        return "text.fp_boundary_error_same_role", same_role_overlap[0]

    overlap = [
        item for item in gold_candidates if _same_text_event_and_trigger(item, pred_item) and _span_overlap(item, pred_item)
    ]
    if overlap:
        return "text.fp_wrong_role_and_boundary", overlap[0]

    if gold_schema_roles and pred_item["role"] not in gold_schema_roles:
        return "text.fp_role_outside_schema", None
    return "text.fp_spurious_argument", None


def _classify_unmatched_text_gold(
    gold_item: Mapping[str, Any],
    pred_candidates: list[dict[str, Any]],
) -> tuple[str, Mapping[str, Any] | None]:
    same_span = [item for item in pred_candidates if _same_text_context(gold_item, item) and _same_span(gold_item, item)]
    if same_span:
        return "text.fn_wrong_role_same_span", same_span[0]

    same_role_overlap = [
        item for item in pred_candidates if _same_text_event_and_trigger(gold_item, item) and item["role"] == gold_item["role"] and _span_overlap(gold_item, item)
    ]
    if same_role_overlap:
        return "text.fn_boundary_error_same_role", same_role_overlap[0]

    overlap = [
        item for item in pred_candidates if _same_text_event_and_trigger(gold_item, item) and _span_overlap(gold_item, item)
    ]
    if overlap:
        return "text.fn_wrong_role_and_boundary", overlap[0]
    return "text.fn_missing_argument", None


def _classify_unmatched_image_pred(
    pred_item: Mapping[str, Any],
    gold_candidates: list[dict[str, Any]],
    *,
    iou_threshold: float,
    iou_low: float,
    cover_high: float,
    purity_low: float,
) -> tuple[str, Mapping[str, Any] | None]:
    wrong_role = _best_image_candidate(
        pred_item,
        gold_candidates,
        require_same_role=False,
        require_different_role=True,
        iou_threshold=iou_threshold,
    )
    if wrong_role is not None:
        return "image.fp_wrong_role_correct_box", wrong_role

    same_role = _best_image_candidate(pred_item, gold_candidates, require_same_role=True)
    if same_role is None:
        return "image.fp_spurious_argument", None

    quality = classify_box_quality(
        pred_item["bbox"],
        same_role["bbox"],
        iou_threshold=iou_threshold,
        iou_low=iou_low,
        cover_high=cover_high,
        purity_low=purity_low,
    )
    return _image_fp_bucket_for_quality(quality), same_role


def _classify_unmatched_image_gold(
    gold_item: Mapping[str, Any],
    pred_candidates: list[dict[str, Any]],
    *,
    iou_threshold: float,
    iou_low: float,
    cover_high: float,
    purity_low: float,
) -> tuple[str, Mapping[str, Any] | None]:
    wrong_role = _best_image_candidate(
        gold_item,
        pred_candidates,
        require_same_role=False,
        require_different_role=True,
        iou_threshold=iou_threshold,
    )
    if wrong_role is not None:
        return "image.fn_wrong_role_correct_box", wrong_role

    same_role = _best_image_candidate(gold_item, pred_candidates, require_same_role=True)
    if same_role is None:
        return "image.fn_missing_argument", None

    quality = classify_box_quality(
        same_role["bbox"],
        gold_item["bbox"],
        iou_threshold=iou_threshold,
        iou_low=iou_low,
        cover_high=cover_high,
        purity_low=purity_low,
    )
    return _image_fn_bucket_for_quality(quality), same_role


def classify_box_quality(
    pred_bbox: list[float],
    gold_bbox: list[float],
    *,
    iou_threshold: float,
    iou_low: float,
    cover_high: float,
    purity_low: float,
) -> str:
    iou = bbox_iou(pred_bbox, gold_bbox)
    if iou >= iou_threshold:
        return "matched"
    if iou <= 0.0:
        return "no_overlap"
    if iou < iou_low:
        return "weak_overlap"

    intersection = _bbox_intersection_area(pred_bbox, gold_bbox)
    pred_area = _bbox_area(pred_bbox)
    gold_area = _bbox_area(gold_bbox)
    coverage = intersection / gold_area if gold_area > 0 else 0.0
    purity = intersection / pred_area if pred_area > 0 else 0.0
    if coverage >= cover_high and purity < purity_low:
        return "box_not_precise_enough_oversized"
    if purity >= cover_high and coverage < purity_low:
        return "box_not_precise_enough_undersized"
    return "box_not_precise_enough_shifted"


def _image_fp_bucket_for_quality(quality: str) -> str:
    return {
        "box_not_precise_enough_oversized": "image.fp_box_not_precise_enough_oversized",
        "box_not_precise_enough_undersized": "image.fp_box_not_precise_enough_undersized",
        "box_not_precise_enough_shifted": "image.fp_box_not_precise_enough_shifted",
        "weak_overlap": "image.fp_weak_overlap_same_role",
        "no_overlap": "image.fp_no_overlap_same_role",
    }.get(quality, "image.fp_spurious_argument")


def _image_fn_bucket_for_quality(quality: str) -> str:
    return {
        "box_not_precise_enough_oversized": "image.fn_box_not_precise_enough_oversized",
        "box_not_precise_enough_undersized": "image.fn_box_not_precise_enough_undersized",
        "box_not_precise_enough_shifted": "image.fn_box_not_precise_enough_shifted",
        "weak_overlap": "image.fn_weak_overlap_same_role",
        "no_overlap": "image.fn_no_overlap_same_role",
    }.get(quality, "image.fn_missing_argument")


def _best_image_candidate(
    item: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    *,
    require_same_role: bool = False,
    require_different_role: bool = False,
    iou_threshold: float | None = None,
) -> Mapping[str, Any] | None:
    best: Mapping[str, Any] | None = None
    best_iou = -1.0
    for candidate in candidates:
        if candidate["event_type"] != item["event_type"]:
            continue
        same_role = candidate["role"] == item["role"]
        if require_same_role and not same_role:
            continue
        if require_different_role and same_role:
            continue
        iou = bbox_iou(candidate["bbox"], item["bbox"])
        if iou_threshold is not None and iou < iou_threshold:
            continue
        if iou > best_iou:
            best = candidate
            best_iou = iou
    return best


def _same_text_event_and_trigger(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        first["event_type"] == second["event_type"]
        and first.get("trigger_start") == second.get("trigger_start")
        and first.get("trigger_end") == second.get("trigger_end")
    )


def _same_text_context(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return _same_text_event_and_trigger(first, second)


def _same_span(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return first.get("start") == second.get("start") and first.get("end") == second.get("end")


def _span_overlap(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    first_start = first.get("start")
    first_end = first.get("end")
    second_start = second.get("start")
    second_end = second.get("end")
    if not all(isinstance(value, int) for value in (first_start, first_end, second_start, second_end)):
        return False
    return max(first_start, second_start) < min(first_end, second_end)


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_intersection_area(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def _extract_invalid_predicted_image_arguments(record: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(record, Mapping):
        return []
    prediction = record.get("prediction")
    if not isinstance(prediction, Mapping):
        prediction = record
    arguments = prediction.get("image_arguments")
    if not isinstance(arguments, list):
        return []
    invalid: list[dict[str, Any]] = []
    for index, argument in enumerate(arguments):
        if not isinstance(argument, Mapping):
            continue
        if normalize_bbox(argument.get("bbox")) is not None:
            continue
        invalid.append(
            {
                "index": index,
                "role": str(argument.get("role") or ""),
                "bbox": argument.get("bbox"),
                "grounding_status": argument.get("grounding_status"),
                "note": "invalid_or_missing_bbox",
            }
        )
    return invalid


def _sample_context(
    gold_sample: Mapping[str, Any],
    pred_record: Mapping[str, Any] | None,
    trace_record: Mapping[str, Any] | None,
    metrics_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    sample_id = get_m2e2_sample_id(gold_sample) or str(gold_sample.get("id") or "")
    return {
        "sample_id": sample_id,
        "gold_event_type": resolve_gold_event_type(gold_sample),
        "predicted_event_type": resolve_predicted_event_type(pred_record),
        "trace": _trace_hints(trace_record),
        "metrics": _metric_hints(metrics_record),
    }


def _trace_hints(trace_record: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(trace_record, Mapping):
        return {}
    stage_c_candidates = _extract_stage_c_candidates(trace_record)
    verifier_output = trace_record.get("verifier_output")
    return {
        "stage_c_candidate_count": len(stage_c_candidates),
        "has_stage_c_candidate": bool(stage_c_candidates),
        "grounding_attempted": bool(trace_record.get("grounding_requests") or trace_record.get("grounding_results")),
        "verifier_verdict": verifier_output.get("verdict") if isinstance(verifier_output, Mapping) else None,
    }


def _metric_hints(metrics_record: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics_record, Mapping):
        return {}
    return {
        key: metrics_record.get(key)
        for key in ("sample_id", "event_type_match", "text_argument", "image_argument", "overall_argument")
        if key in metrics_record
    }


def _extract_stage_c_candidates(trace_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    stage_c_output = trace_record.get("stage_c_output")
    if isinstance(stage_c_output, Mapping):
        candidates = stage_c_output.get("image_arguments")
        if isinstance(candidates, list):
            return [dict(item) for item in candidates if isinstance(item, Mapping)]
    if isinstance(stage_c_output, list):
        return [dict(item) for item in stage_c_output if isinstance(item, Mapping)]
    return []


def _add_counter(
    bucket: str,
    counters: Counter[str],
    examples: dict[str, list[dict[str, Any]]],
    context: Mapping[str, Any],
    sample_limit: int,
    *,
    gold: Mapping[str, Any] | None = None,
    pred: Mapping[str, Any] | None = None,
) -> None:
    counters[bucket] += 1
    if len(examples[bucket]) >= sample_limit:
        return
    examples[bucket].append(
        {
            "sample_id": context.get("sample_id"),
            "gold_event_type": context.get("gold_event_type"),
            "predicted_event_type": context.get("predicted_event_type"),
            "gold": _summarize_arg(gold),
            "pred": _summarize_arg(pred),
            "trace": context.get("trace") or {},
            "metrics": context.get("metrics") or {},
        }
    )


def _summarize_arg(arg: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(arg, Mapping):
        return None
    result = {
        "event_type": arg.get("event_type"),
        "role": arg.get("role"),
        "raw_role": arg.get("raw_role"),
    }
    if "start" in arg or "end" in arg:
        result.update({"start": arg.get("start"), "end": arg.get("end")})
    if "bbox" in arg:
        result["bbox"] = arg.get("bbox")
    if "grounding_status" in arg:
        result["grounding_status"] = arg.get("grounding_status")
    if "note" in arg:
        result["note"] = arg.get("note")
    return result


def _print_breakdown(title: str, summary: Mapping[str, int]) -> None:
    print(title)
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()
    gold_samples = load_json_or_jsonl(args.gold)
    prediction_records = load_json_or_jsonl(args.pred)
    trace_records = load_json_or_jsonl(args.trace) if args.trace else []
    metrics_records = load_json_or_jsonl(args.per_sample_metrics) if args.per_sample_metrics else []

    report = analyze_error_breakdown(
        gold_samples,
        prediction_records,
        trace_records=trace_records,
        per_sample_metrics_records=metrics_records,
        iou_threshold=float(args.iou_threshold),
        iou_low=float(args.iou_low),
        cover_high=float(args.cover_high),
        purity_low=float(args.purity_low),
        sample_limit=max(0, int(args.sample_limit)),
    )

    _print_breakdown("TEXT ERROR BREAKDOWN", report["summary"]["text"])
    print()
    _print_breakdown("IMAGE ERROR BREAKDOWN", report["summary"]["image"])

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
