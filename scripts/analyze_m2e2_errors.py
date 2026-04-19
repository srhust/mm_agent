from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mm_event_agent.m2e2_adapter import get_m2e2_sample_id
from scripts.score_m2e2_current import (
    bbox_iou,
    canonicalize_role,
    extract_gold_image_arguments,
    extract_gold_text_argument_tuples,
    extract_predicted_image_arguments,
    extract_predicted_text_argument_tuples,
    load_json_or_jsonl,
    match_image_arguments,
    resolve_gold_event_type,
    resolve_predicted_event_type,
)

ERROR_CATEGORIES = (
    "event_type_mismatch",
    "text_arg_missing",
    "text_arg_extra",
    "text_arg_boundary_overlong",
    "text_arg_role_mismatch",
    "image_arg_missing",
    "image_arg_extra",
    "image_arg_iou_low",
    "image_arg_generic_weak_place",
    "image_arg_same_role_instance_shortfall",
    "grounding_failed_after_stage_c_candidate",
    "stage_c_missing_candidate",
    "verifier_false_reject",
    "verifier_false_accept",
)

GENERIC_WEAK_PLACE_TERMS = (
    "area",
    "scene",
    "location",
    "place",
    "street",
    "road",
    "outside",
    "outdoor",
    "building",
    "site",
    "market area",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze current M2E2 evaluation outputs and write structured error artifacts."
    )
    parser.add_argument("--gold", required=True, help="Path to gold M2E2 JSON or JSONL samples.")
    parser.add_argument("--pred", required=True, help="Path to predictions.jsonl.")
    parser.add_argument("--trace", required=True, help="Path to trace.jsonl.")
    parser.add_argument("--per-sample-metrics", help="Optional path to per_sample_metrics.jsonl.")
    parser.add_argument(
        "--image-iou",
        type=float,
        default=0.5,
        help="IoU threshold for image matching. Default: 0.5",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional directory for analysis artifacts. Defaults to the prediction file directory.",
    )
    return parser.parse_args()


def index_records_by_id(
    records: Iterable[Mapping[str, Any]],
    *,
    id_keys: tuple[str, ...] = ("sample_id", "id"),
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = ""
        for key in id_keys:
            value = record.get(key)
            if value is not None and str(value).strip():
                sample_id = str(value).strip()
                break
        if sample_id:
            indexed[sample_id] = dict(record)
    return indexed


def text_tuple_to_dict(item: tuple[Any, ...]) -> dict[str, Any]:
    if len(item) == 6:
        event_type, trigger_start, trigger_end, role, start, end = item
    elif len(item) == 4:
        event_type, role, start, end = item
        trigger_start = None
        trigger_end = None
    else:
        return {
            "event_type": "",
            "trigger_start": None,
            "trigger_end": None,
            "role": "",
            "start": None,
            "end": None,
        }
    return {
        "event_type": str(event_type or ""),
        "trigger_start": trigger_start if isinstance(trigger_start, int) else None,
        "trigger_end": trigger_end if isinstance(trigger_end, int) else None,
        "role": str(role or ""),
        "start": start if isinstance(start, int) else None,
        "end": end if isinstance(end, int) else None,
    }


def summarize_text_arg(arg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": str(arg.get("role") or ""),
        "start": arg.get("start"),
        "end": arg.get("end"),
    }


def summarize_image_arg(arg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": str(arg.get("role") or ""),
        "bbox": list(arg.get("bbox", [])) if isinstance(arg.get("bbox"), list) else None,
    }


def analyze_predictions(
    gold_samples: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    trace_records: list[dict[str, Any]],
    *,
    per_sample_metrics_records: list[dict[str, Any]] | None = None,
    image_iou: float = 0.5,
) -> dict[str, Any]:
    predictions_by_id = index_records_by_id(prediction_records, id_keys=("id", "sample_id"))
    traces_by_id = index_records_by_id(trace_records)
    metrics_by_id = index_records_by_id(per_sample_metrics_records or [])

    category_counts: Counter[str] = Counter()
    event_type_counts: dict[str, Counter[str]] = {}
    role_counts: dict[str, Counter[str]] = {}
    stage_cause_counts: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []

    for gold_sample in gold_samples:
        case = analyze_sample(
            gold_sample,
            predictions_by_id.get(get_m2e2_sample_id(gold_sample)),
            traces_by_id.get(get_m2e2_sample_id(gold_sample)),
            metrics_by_id.get(get_m2e2_sample_id(gold_sample)),
            image_iou=image_iou,
        )
        if not case["triggered_error_categories"]:
            continue
        cases.append(case)
        event_type = case["gold_summary"]["event_type"] or "(missing)"
        event_bucket = event_type_counts.setdefault(event_type, Counter())
        for category in case["triggered_error_categories"]:
            category_counts[category] += 1
            event_bucket[category] += 1
        for role in case["roles_involved"]:
            role_bucket = role_counts.setdefault(role or "(missing)", Counter())
            for category in case["triggered_error_categories"]:
                role_bucket[category] += 1
        for cause in case["likely_source_stages"]:
            stage_cause_counts[cause] += 1

    return {
        "error_summary": {
            "n_gold_samples": len(gold_samples),
            "n_problem_cases": len(cases),
            "counts_per_category": {category: int(category_counts.get(category, 0)) for category in ERROR_CATEGORIES},
            "counts_per_event_type": {
                event_type: {category: int(counter.get(category, 0)) for category in ERROR_CATEGORIES}
                for event_type, counter in sorted(event_type_counts.items())
            },
            "counts_per_role": {
                role: {category: int(counter.get(category, 0)) for category in ERROR_CATEGORIES}
                for role, counter in sorted(role_counts.items())
            },
            "counts_per_stage_attributed_cause": dict(sorted(stage_cause_counts.items())),
        },
        "error_cases": cases,
    }


def analyze_sample(
    gold_sample: Mapping[str, Any],
    prediction_record: Mapping[str, Any] | None,
    trace_record: Mapping[str, Any] | None,
    metrics_record: Mapping[str, Any] | None,
    *,
    image_iou: float,
) -> dict[str, Any]:
    sample_id = get_m2e2_sample_id(gold_sample)
    gold_event_type = resolve_gold_event_type(gold_sample)
    predicted_event_type = resolve_predicted_event_type(prediction_record)

    gold_text = [text_tuple_to_dict(item) for item in extract_gold_text_argument_tuples(gold_sample, ignore_trigger=False)]
    pred_text = [text_tuple_to_dict(item) for item in extract_predicted_text_argument_tuples(prediction_record, ignore_trigger=False)]
    gold_image = extract_gold_image_arguments(gold_sample)
    pred_image = extract_predicted_image_arguments(prediction_record)
    image_tp, image_matches = match_image_arguments(pred_image, gold_image, image_iou=image_iou)

    categories: set[str] = set()
    likely_sources: set[str] = set()
    roles_involved: set[str] = set()
    notes: list[str] = []

    if gold_event_type and predicted_event_type != gold_event_type:
        categories.add("event_type_mismatch")
        likely_sources.add("stage_a_event_type_selection")
        notes.append(f"predicted event type {predicted_event_type or '(missing)'} does not match gold {gold_event_type}.")

    text_analysis = analyze_text_argument_errors(gold_text, pred_text)
    categories.update(text_analysis["categories"])
    likely_sources.update(text_analysis["likely_sources"])
    roles_involved.update(text_analysis["roles"])
    notes.extend(text_analysis["notes"])

    image_analysis = analyze_image_argument_errors(
        gold_image,
        pred_image,
        image_matches=image_matches,
        trace_record=trace_record,
        image_iou=image_iou,
    )
    categories.update(image_analysis["categories"])
    likely_sources.update(image_analysis["likely_sources"])
    roles_involved.update(image_analysis["roles"])
    notes.extend(image_analysis["notes"])

    attribution = analyze_trace_attribution(
        gold_sample,
        gold_text,
        pred_text,
        gold_image,
        pred_image,
        trace_record=trace_record,
        image_iou=image_iou,
    )
    categories.update(attribution["categories"])
    likely_sources.update(attribution["likely_sources"])
    roles_involved.update(attribution["roles"])
    notes.extend(attribution["notes"])

    verifier_categories = analyze_verifier_pattern(
        gold_event_type=gold_event_type,
        predicted_event_type=predicted_event_type,
        gold_text=gold_text,
        pred_text=pred_text,
        gold_image=gold_image,
        pred_image=pred_image,
        image_tp=image_tp,
        trace_record=trace_record,
    )
    categories.update(verifier_categories["categories"])
    likely_sources.update(verifier_categories["likely_sources"])
    notes.extend(verifier_categories["notes"])

    if not categories:
        return {
            "id": sample_id,
            "gold_summary": build_gold_summary(gold_event_type, gold_text, gold_image),
            "predicted_summary": build_pred_summary(predicted_event_type, pred_text, pred_image),
            "triggered_error_categories": [],
            "likely_source_stages": [],
            "roles_involved": [],
            "analysis_note": "",
            "metrics": dict(metrics_record) if isinstance(metrics_record, Mapping) else {},
        }

    return {
        "id": sample_id,
        "gold_summary": build_gold_summary(gold_event_type, gold_text, gold_image),
        "predicted_summary": build_pred_summary(predicted_event_type, pred_text, pred_image),
        "triggered_error_categories": sorted(categories),
        "likely_source_stages": sorted(likely_sources),
        "roles_involved": sorted(role for role in roles_involved if role),
        "analysis_note": " ".join(deduplicate_preserve_order(notes)),
        "metrics": dict(metrics_record) if isinstance(metrics_record, Mapping) else {},
    }


def analyze_text_argument_errors(
    gold_text: list[dict[str, Any]],
    pred_text: list[dict[str, Any]],
) -> dict[str, Any]:
    categories: set[str] = set()
    likely_sources: set[str] = set()
    roles: set[str] = set()
    notes: list[str] = []

    unmatched_gold, unmatched_pred = remove_exact_text_matches(gold_text, pred_text)

    role_pairs = find_text_role_mismatch_pairs(unmatched_gold, unmatched_pred)
    if role_pairs:
        categories.add("text_arg_role_mismatch")
        likely_sources.add("stage_b_role_assignment")
        for gold_arg, pred_arg in role_pairs:
            roles.add(gold_arg["role"])
            roles.add(pred_arg["role"])
            notes.append(
                f"text role mismatch around span {gold_arg['start']}-{gold_arg['end']}: gold {gold_arg['role']} vs predicted {pred_arg['role']}."
            )

    overlong_pairs = find_text_overlong_pairs(unmatched_gold, unmatched_pred)
    if overlong_pairs:
        categories.add("text_arg_boundary_overlong")
        likely_sources.add("stage_b_text_boundary")
        for gold_arg, pred_arg in overlong_pairs:
            roles.add(gold_arg["role"])
            notes.append(
                f"text argument for role {gold_arg['role']} is overlong: predicted {pred_arg['start']}-{pred_arg['end']} vs gold {gold_arg['start']}-{gold_arg['end']}."
            )

    if unmatched_gold:
        categories.add("text_arg_missing")
        likely_sources.add("stage_b_argument_recall")
        for arg in unmatched_gold:
            roles.add(arg["role"])
        notes.append(f"missing {len(unmatched_gold)} text argument(s) under strict token-span matching.")

    if unmatched_pred:
        categories.add("text_arg_extra")
        likely_sources.add("stage_b_argument_precision")
        for arg in unmatched_pred:
            roles.add(arg["role"])
        notes.append(f"predicted {len(unmatched_pred)} extra text argument(s) under strict token-span matching.")

    return {
        "categories": categories,
        "likely_sources": likely_sources,
        "roles": roles,
        "notes": notes,
    }


def remove_exact_text_matches(
    gold_text: list[dict[str, Any]],
    pred_text: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unmatched_gold = list(gold_text)
    unmatched_pred = list(pred_text)
    used_pred_indices: set[int] = set()
    filtered_gold: list[dict[str, Any]] = []
    for gold_arg in unmatched_gold:
        matched_index = None
        for pred_index, pred_arg in enumerate(unmatched_pred):
            if pred_index in used_pred_indices:
                continue
            if gold_arg == pred_arg:
                matched_index = pred_index
                break
        if matched_index is None:
            filtered_gold.append(gold_arg)
            continue
        used_pred_indices.add(matched_index)
    filtered_pred = [arg for index, arg in enumerate(unmatched_pred) if index not in used_pred_indices]
    return filtered_gold, filtered_pred


def find_text_role_mismatch_pairs(
    gold_text: list[dict[str, Any]],
    pred_text: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_pred_indices: set[int] = set()
    for gold_arg in gold_text:
        for pred_index, pred_arg in enumerate(pred_text):
            if pred_index in used_pred_indices:
                continue
            if (
                gold_arg["event_type"] == pred_arg["event_type"]
                and gold_arg["trigger_start"] == pred_arg["trigger_start"]
                and gold_arg["trigger_end"] == pred_arg["trigger_end"]
                and gold_arg["start"] == pred_arg["start"]
                and gold_arg["end"] == pred_arg["end"]
                and gold_arg["role"] != pred_arg["role"]
            ):
                pairs.append((gold_arg, pred_arg))
                used_pred_indices.add(pred_index)
                break
    return pairs


def find_text_overlong_pairs(
    gold_text: list[dict[str, Any]],
    pred_text: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_pred_indices: set[int] = set()
    for gold_arg in gold_text:
        for pred_index, pred_arg in enumerate(pred_text):
            if pred_index in used_pred_indices:
                continue
            if (
                gold_arg["event_type"] == pred_arg["event_type"]
                and gold_arg["trigger_start"] == pred_arg["trigger_start"]
                and gold_arg["trigger_end"] == pred_arg["trigger_end"]
                and gold_arg["role"] == pred_arg["role"]
                and isinstance(gold_arg["start"], int)
                and isinstance(gold_arg["end"], int)
                and isinstance(pred_arg["start"], int)
                and isinstance(pred_arg["end"], int)
                and pred_arg["start"] <= gold_arg["start"]
                and pred_arg["end"] >= gold_arg["end"]
                and (pred_arg["start"] < gold_arg["start"] or pred_arg["end"] > gold_arg["end"])
            ):
                pairs.append((gold_arg, pred_arg))
                used_pred_indices.add(pred_index)
                break
    return pairs


def analyze_image_argument_errors(
    gold_image: list[dict[str, Any]],
    pred_image: list[dict[str, Any]],
    *,
    image_matches: list[dict[str, Any]],
    trace_record: Mapping[str, Any] | None,
    image_iou: float,
) -> dict[str, Any]:
    categories: set[str] = set()
    likely_sources: set[str] = set()
    roles: set[str] = set()
    notes: list[str] = []

    matched_gold_indices = {
        int(match["matched_gold_index"])
        for match in image_matches
        if match.get("matched") and isinstance(match.get("matched_gold_index"), int)
    }
    unmatched_gold = [item for index, item in enumerate(gold_image) if index not in matched_gold_indices]
    unmatched_pred = [
        pred_image[index]
        for index, match in enumerate(image_matches)
        if not match.get("matched") and index < len(pred_image)
    ]

    if unmatched_gold:
        categories.add("image_arg_missing")
        likely_sources.add("image_argument_recall")
        for item in unmatched_gold:
            roles.add(item["role"])
        notes.append(f"missing {len(unmatched_gold)} grounded image argument(s).")

    if unmatched_pred:
        categories.add("image_arg_extra")
        likely_sources.add("image_argument_precision")
        for item in unmatched_pred:
            roles.add(item["role"])
        notes.append(f"predicted {len(unmatched_pred)} extra grounded image argument(s).")

    low_iou_roles = find_low_iou_image_roles(unmatched_gold, unmatched_pred, image_iou=image_iou)
    if low_iou_roles:
        categories.add("image_arg_iou_low")
        likely_sources.add("grounding_box_quality")
        roles.update(low_iou_roles)
        notes.append(f"found low-IoU image matches below threshold {image_iou}.")

    weak_place_roles = find_weak_place_hallucination_roles(trace_record, gold_image, pred_image)
    if weak_place_roles:
        categories.add("image_arg_generic_weak_place")
        likely_sources.add("stage_c_weak_place_proposal")
        roles.update(weak_place_roles)
        notes.append("generic weak Place proposal survived into final grounded predictions.")

    shortfall_roles = find_same_role_image_shortfall_roles(gold_image, pred_image)
    if shortfall_roles:
        categories.add("image_arg_same_role_instance_shortfall")
        likely_sources.add("image_multi_instance_recall")
        roles.update(shortfall_roles)
        notes.append("fewer grounded same-role image instances were predicted than gold requires.")

    return {
        "categories": categories,
        "likely_sources": likely_sources,
        "roles": roles,
        "notes": notes,
    }


def find_low_iou_image_roles(
    unmatched_gold: list[dict[str, Any]],
    unmatched_pred: list[dict[str, Any]],
    *,
    image_iou: float,
) -> set[str]:
    roles: set[str] = set()
    for gold_item in unmatched_gold:
        for pred_item in unmatched_pred:
            if gold_item["event_type"] != pred_item["event_type"]:
                continue
            if gold_item["role"] != pred_item["role"]:
                continue
            overlap = bbox_iou(gold_item["bbox"], pred_item["bbox"])
            if 0.0 < overlap < image_iou:
                roles.add(gold_item["role"])
    return roles


def find_same_role_image_shortfall_roles(
    gold_image: list[dict[str, Any]],
    pred_image: list[dict[str, Any]],
) -> set[str]:
    gold_counts = Counter(item["role"] for item in gold_image)
    pred_counts = Counter(item["role"] for item in pred_image)
    return {
        role
        for role, gold_count in gold_counts.items()
        if gold_count > 1 and pred_counts.get(role, 0) < gold_count
    }


def find_weak_place_hallucination_roles(
    trace_record: Mapping[str, Any] | None,
    gold_image: list[dict[str, Any]],
    pred_image: list[dict[str, Any]],
) -> set[str]:
    if not isinstance(trace_record, Mapping):
        return set()
    stage_c_candidates = extract_stage_c_candidates(trace_record)
    predicted_place_count = sum(1 for item in pred_image if item["role"] == "Place")
    gold_place_count = sum(1 for item in gold_image if item["role"] == "Place")
    roles: set[str] = set()
    for candidate in stage_c_candidates:
        if candidate.get("role") != "Place":
            continue
        if not is_generic_weak_place_label(candidate.get("label")):
            continue
        if predicted_place_count > gold_place_count:
            roles.add("Place")
    return roles


def analyze_trace_attribution(
    gold_sample: Mapping[str, Any],
    gold_text: list[dict[str, Any]],
    pred_text: list[dict[str, Any]],
    gold_image: list[dict[str, Any]],
    pred_image: list[dict[str, Any]],
    *,
    trace_record: Mapping[str, Any] | None,
    image_iou: float,
) -> dict[str, Any]:
    categories: set[str] = set()
    likely_sources: set[str] = set()
    roles: set[str] = set()
    notes: list[str] = []

    if not isinstance(trace_record, Mapping):
        return {"categories": categories, "likely_sources": likely_sources, "roles": roles, "notes": notes}

    stage_b_output = trace_record.get("stage_b_output")
    if isinstance(stage_b_output, Mapping):
        stage_b_arguments = stage_b_output.get("text_arguments")
        if isinstance(stage_b_arguments, list):
            text_value_lookup = extract_gold_text_value_lookup(gold_sample)
            final_exact_values = {value for value in text_value_lookup if value in predicted_text_values(gold_sample, pred_text)}
            for argument in stage_b_arguments:
                if not isinstance(argument, Mapping):
                    continue
                role = str(argument.get("role") or "")
                text = str(argument.get("text") or "").strip()
                if not role or not text:
                    continue
                for gold_role, gold_text_value in text_value_lookup.items():
                    if role != gold_role:
                        continue
                    if text != gold_text_value and gold_text_value and gold_text_value in text and gold_text_value in final_exact_values:
                        notes.append(
                            f"stage_b proposed broader text for role {role} but final prediction normalized it to the stricter gold span."
                        )
                        break

    stage_c_candidates = extract_stage_c_candidates(trace_record)
    grounding_results = trace_record.get("grounding_results")
    grounding_result_list = list(grounding_results) if isinstance(grounding_results, list) else []
    final_role_counts = Counter(item["role"] for item in pred_image)

    for gold_role, gold_count in Counter(item["role"] for item in gold_image).items():
        predicted_count = final_role_counts.get(gold_role, 0)
        if predicted_count >= gold_count:
            continue
        candidate_count = sum(1 for item in stage_c_candidates if item.get("role") == gold_role)
        grounded_count = sum(
            1
            for item in grounding_result_list
            if canonicalize_role(item.get("role")) == gold_role and str(item.get("grounding_status") or "") == "grounded"
        )
        if candidate_count < gold_count:
            categories.add("stage_c_missing_candidate")
            likely_sources.add("stage_c_missing_candidate")
            roles.add(gold_role)
            notes.append(f"stage_c proposed only {candidate_count} candidate(s) for gold image role {gold_role}, below the gold count {gold_count}.")
        elif grounded_count < min(candidate_count, gold_count):
            categories.add("grounding_failed_after_stage_c_candidate")
            likely_sources.add("grounding_failed_after_stage_c_candidate")
            roles.add(gold_role)
            notes.append(f"stage_c proposed role {gold_role}, but grounding did not recover enough grounded boxes.")

    for candidate in stage_c_candidates:
        if candidate.get("role") == "Place" and is_generic_weak_place_label(candidate.get("label")):
            if final_role_counts.get("Place", 0) <= Counter(item["role"] for item in gold_image).get("Place", 0):
                notes.append("generic weak Place candidate appeared in stage_c but did not survive to the final grounded prediction.")
                break

    return {
        "categories": categories,
        "likely_sources": likely_sources,
        "roles": roles,
        "notes": notes,
    }


def extract_gold_text_value_lookup(gold_sample: Mapping[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    mentions = gold_sample.get("text_event_mentions")
    if not isinstance(mentions, list):
        return values
    for mention in mentions:
        if not isinstance(mention, Mapping):
            continue
        arguments = mention.get("arguments")
        if not isinstance(arguments, list):
            continue
        for argument in arguments:
            if not isinstance(argument, Mapping):
                continue
            role = canonicalize_role(argument.get("role"))
            text = str(argument.get("text") or "").strip()
            if role and text and role not in values:
                values[role] = text
    return values


def predicted_text_values(gold_sample: Mapping[str, Any], pred_text: list[dict[str, Any]]) -> set[str]:
    words = gold_sample.get("words")
    tokens = list(words) if isinstance(words, list) else []
    values: set[str] = set()
    for arg in pred_text:
        start = arg.get("start")
        end = arg.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(tokens):
            values.add(" ".join(str(token) for token in tokens[start:end]).strip())
    return values


def extract_stage_c_candidates(trace_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    stage_c_output = trace_record.get("stage_c_output")
    if not isinstance(stage_c_output, Mapping):
        return []
    candidates = stage_c_output.get("image_arguments")
    if not isinstance(candidates, list):
        return []
    result: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        role = canonicalize_role(item.get("role"))
        label = str(item.get("label") or "").strip()
        result.append({"role": role, "label": label})
    return result


def is_generic_weak_place_label(label: Any) -> bool:
    value = str(label or "").strip().lower()
    if not value:
        return False
    return any(term in value for term in GENERIC_WEAK_PLACE_TERMS)


def analyze_verifier_pattern(
    *,
    gold_event_type: str,
    predicted_event_type: str,
    gold_text: list[dict[str, Any]],
    pred_text: list[dict[str, Any]],
    gold_image: list[dict[str, Any]],
    pred_image: list[dict[str, Any]],
    image_tp: int,
    trace_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    categories: set[str] = set()
    likely_sources: set[str] = set()
    notes: list[str] = []
    if not isinstance(trace_record, Mapping):
        return {"categories": categories, "likely_sources": likely_sources, "notes": notes}
    verifier_output = trace_record.get("verifier_output")
    if not isinstance(verifier_output, Mapping):
        return {"categories": categories, "likely_sources": likely_sources, "notes": notes}

    verified = bool(verifier_output.get("verified"))
    text_correct = not remove_exact_text_matches(gold_text, pred_text)[0] and not remove_exact_text_matches(gold_text, pred_text)[1]
    image_correct = image_tp == len(gold_image) == len(pred_image)
    event_correct = bool(gold_event_type) and predicted_event_type == gold_event_type

    if not verified and event_correct and text_correct and image_correct:
        categories.add("verifier_false_reject")
        likely_sources.add("verifier")
        notes.append("verifier rejected a sample that appears correct under the current scoring assumptions.")
    if verified and not (event_correct and text_correct and image_correct):
        categories.add("verifier_false_accept")
        likely_sources.add("verifier")
        notes.append("verifier accepted a sample that still contains scoring-visible errors.")

    return {"categories": categories, "likely_sources": likely_sources, "notes": notes}


def build_gold_summary(
    event_type: str,
    gold_text: list[dict[str, Any]],
    gold_image: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "text_arguments": [summarize_text_arg(item) for item in gold_text],
        "image_arguments": [summarize_image_arg(item) for item in gold_image],
    }


def build_pred_summary(
    event_type: str,
    pred_text: list[dict[str, Any]],
    pred_image: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "text_arguments": [summarize_text_arg(item) for item in pred_text],
        "image_arguments": [summarize_image_arg(item) for item in pred_image],
    }


def deduplicate_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def write_analysis_outputs(
    output_dir: str | Path,
    *,
    error_summary: Mapping[str, Any],
    error_cases: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_path = target_dir / "error_summary.json"
    cases_path = target_dir / "error_cases.jsonl"
    summary_path.write_text(json.dumps(dict(error_summary), ensure_ascii=False, indent=2), encoding="utf-8")
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in error_cases:
            handle.write(json.dumps(dict(case), ensure_ascii=False) + "\n")
    return {
        "error_summary": str(summary_path),
        "error_cases": str(cases_path),
    }


def main() -> None:
    args = parse_args()
    gold_samples = load_json_or_jsonl(args.gold)
    prediction_records = load_json_or_jsonl(args.pred)
    trace_records = load_json_or_jsonl(args.trace)
    per_sample_metrics = load_json_or_jsonl(args.per_sample_metrics) if args.per_sample_metrics else []

    report = analyze_predictions(
        gold_samples,
        prediction_records,
        trace_records,
        per_sample_metrics_records=per_sample_metrics,
        image_iou=float(args.image_iou),
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.pred).resolve().parent
    written_files = write_analysis_outputs(
        output_dir,
        error_summary=report["error_summary"],
        error_cases=report["error_cases"],
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "files": written_files,
                "n_problem_cases": report["error_summary"]["n_problem_cases"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
