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

from mm_event_agent.ontology import get_supported_event_types


_EVENT_TYPE_EXPLICIT_ALIASES = {
    "Justice:ArrestJail": "Justice:Arrest-Jail",
    "Contact:PhoneWrite": "Contact:Phone-Write",
    "Transaction:TransferMoney": "Transaction:Transfer-Money",
    "Arrest": "Justice:Arrest-Jail",
    "Writing": "Contact:Phone-Write",
}

_ROLE_EXPLICIT_ALIASES = {
    "Detainee": "Person",
    "Suspect": "Person",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score current project M2E2 predictions against gold JSON/JSONL samples."
    )
    parser.add_argument("--gold", required=True, help="Path to M2E2 gold JSON or JSONL.")
    parser.add_argument("--pred", required=True, help="Path to current project predictions.jsonl.")
    parser.add_argument(
        "--image-iou",
        type=float,
        default=0.5,
        help="IoU threshold for image argument matching. Default: 0.5",
    )
    parser.add_argument(
        "--ignore-trigger",
        action="store_true",
        help="Ignore trigger span in text argument matching.",
    )
    parser.add_argument("--save", help="Optional JSON file path for the full score report.")
    parser.add_argument(
        "--comparison-preview",
        type=int,
        default=5,
        help="Number of per-sample comparisons to include in the report preview.",
    )
    return parser.parse_args()


def canonicalize_event_type(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw in _EVENT_TYPE_EXPLICIT_ALIASES:
        return _EVENT_TYPE_EXPLICIT_ALIASES[raw]
    supported = get_supported_event_types()
    if raw in supported:
        return raw

    normalized_key = _normalize_key(raw)
    canonical_lookup = {_normalize_key(event_type): event_type for event_type in supported}
    canonical_lookup.update(
        {_normalize_key(alias): target for alias, target in _EVENT_TYPE_EXPLICIT_ALIASES.items()}
    )
    return canonical_lookup.get(normalized_key, raw)


def canonicalize_role(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw in _ROLE_EXPLICIT_ALIASES:
        return _ROLE_EXPLICIT_ALIASES[raw]
    normalized_key = _normalize_key(raw)
    normalized_aliases = {
        _normalize_key(alias): target for alias, target in _ROLE_EXPLICIT_ALIASES.items()
    }
    if normalized_key in normalized_aliases:
        return normalized_aliases[normalized_key]
    return raw


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            payload = line.strip()
            if not payload:
                continue
            data = json.loads(payload)
            if isinstance(data, dict):
                records.append(data)
        return records

    data = json.loads(text)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("samples", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    raise ValueError(f"Unsupported JSON payload in {input_path}")


def index_predictions_by_id(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record.get("id") or "").strip()
        if not sample_id:
            continue
        indexed[sample_id] = dict(record)
    return indexed


def resolve_gold_event_type(sample: Mapping[str, Any]) -> str:
    meta = sample.get("meta")
    if isinstance(meta, Mapping):
        event_type = canonicalize_event_type(meta.get("crossmedia_event_type"))
        if event_type:
            return event_type
    event_type = canonicalize_event_type(sample.get("event_type"))
    if event_type:
        return event_type
    mentions = sample.get("text_event_mentions")
    if isinstance(mentions, list):
        for mention in mentions:
            if isinstance(mention, Mapping):
                event_type = canonicalize_event_type(mention.get("event_type"))
                if event_type:
                    return event_type
    image_event = sample.get("image_event")
    if isinstance(image_event, Mapping):
        event_type = canonicalize_event_type(image_event.get("event_type"))
        if event_type:
            return event_type
    return ""


def resolve_predicted_event_type(record: Mapping[str, Any] | None) -> str:
    if not isinstance(record, Mapping):
        return ""
    prediction = record.get("prediction")
    if isinstance(prediction, Mapping):
        return canonicalize_event_type(prediction.get("event_type"))
    return canonicalize_event_type(record.get("event_type"))


def extract_predicted_trigger(record: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    if not isinstance(record, Mapping):
        return (None, None)
    prediction = record.get("prediction")
    if not isinstance(prediction, Mapping):
        prediction = record
    trigger = prediction.get("trigger")
    if not isinstance(trigger, Mapping):
        return (None, None)
    start = trigger.get("start")
    end = trigger.get("end")
    return (start if isinstance(start, int) else None, end if isinstance(end, int) else None)


def extract_gold_text_argument_tuples(
    sample: Mapping[str, Any],
    *,
    ignore_trigger: bool,
) -> list[tuple[Any, ...]]:
    tuples: list[tuple[Any, ...]] = []
    mentions = sample.get("text_event_mentions")
    if not isinstance(mentions, list):
        return tuples
    for mention in mentions:
        if not isinstance(mention, Mapping):
            continue
        event_type = canonicalize_event_type(mention.get("event_type") or resolve_gold_event_type(sample))
        trigger = mention.get("trigger")
        trigger_start = trigger.get("start") if isinstance(trigger, Mapping) and isinstance(trigger.get("start"), int) else None
        trigger_end = trigger.get("end") if isinstance(trigger, Mapping) and isinstance(trigger.get("end"), int) else None
        arguments = mention.get("arguments")
        if not isinstance(arguments, list):
            continue
        for argument in arguments:
            if not isinstance(argument, Mapping):
                continue
            role = canonicalize_role(argument.get("role"))
            start = argument.get("start")
            end = argument.get("end")
            if not role or not isinstance(start, int) or not isinstance(end, int):
                continue
            if ignore_trigger:
                tuples.append((event_type, role, start, end))
            else:
                tuples.append((event_type, trigger_start, trigger_end, role, start, end))
    return tuples


def extract_predicted_text_argument_tuples(
    record: Mapping[str, Any] | None,
    *,
    ignore_trigger: bool,
) -> list[tuple[Any, ...]]:
    if not isinstance(record, Mapping):
        return []
    prediction = record.get("prediction")
    if not isinstance(prediction, Mapping):
        prediction = record
    event_type = canonicalize_event_type(prediction.get("event_type"))
    trigger_start, trigger_end = extract_predicted_trigger(record)
    arguments = prediction.get("text_arguments")
    if not isinstance(arguments, list):
        return []
    tuples: list[tuple[Any, ...]] = []
    for argument in arguments:
        if not isinstance(argument, Mapping):
            continue
        role = canonicalize_role(argument.get("role"))
        start = argument.get("start")
        end = argument.get("end")
        if not role or not isinstance(start, int) or not isinstance(end, int):
            continue
        if ignore_trigger:
            tuples.append((event_type, role, start, end))
        else:
            tuples.append((event_type, trigger_start, trigger_end, role, start, end))
    return tuples


def extract_gold_image_arguments(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    arguments = sample.get("image_arguments_flat")
    extracted: list[dict[str, Any]] = []
    if isinstance(arguments, list):
        for argument in arguments:
            normalized = _normalize_gold_image_argument(argument)
            if normalized is not None:
                extracted.append(normalized)
        if extracted:
            return extracted

    image_event = sample.get("image_event")
    if not isinstance(image_event, Mapping):
        return []
    event_type = canonicalize_event_type(image_event.get("event_type") or resolve_gold_event_type(sample))
    role_map = image_event.get("role")
    if not isinstance(role_map, Mapping):
        return []
    for role, boxes in role_map.items():
        if not isinstance(boxes, list):
            continue
        for item in boxes:
            if not isinstance(item, Mapping):
                continue
            bbox = normalize_bbox(item.get("bbox"))
            if bbox is None:
                continue
            extracted.append(
                {
                    "event_type": event_type,
                    "role": canonicalize_role(role),
                    "bbox": bbox,
                }
            )
    return extracted


def extract_predicted_image_arguments(record: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(record, Mapping):
        return []
    prediction = record.get("prediction")
    if not isinstance(prediction, Mapping):
        prediction = record
    event_type = canonicalize_event_type(prediction.get("event_type"))
    arguments = prediction.get("image_arguments")
    if not isinstance(arguments, list):
        return []
    extracted: list[dict[str, Any]] = []
    for argument in arguments:
        if not isinstance(argument, Mapping):
            continue
        role = canonicalize_role(argument.get("role"))
        bbox = normalize_bbox(argument.get("bbox"))
        if not role or bbox is None:
            continue
        extracted.append(
            {
                "event_type": event_type,
                "role": role,
                "bbox": bbox,
            }
        )
    return extracted


def normalize_bbox(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    normalized: list[float] = []
    for value in bbox:
        try:
            normalized.append(float(value))
        except (TypeError, ValueError):
            return None
    return normalized


def bbox_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def match_image_arguments(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    *,
    image_iou: float,
) -> tuple[int, list[dict[str, Any]]]:
    matches: list[dict[str, Any]] = []
    unmatched_gold = set(range(len(gold)))
    tp = 0
    for pred_index, pred_item in enumerate(predicted):
        best_gold_index: int | None = None
        best_iou = 0.0
        for gold_index in unmatched_gold:
            gold_item = gold[gold_index]
            if pred_item["event_type"] != gold_item["event_type"]:
                continue
            if pred_item["role"] != gold_item["role"]:
                continue
            overlap = bbox_iou(pred_item["bbox"], gold_item["bbox"])
            if overlap > image_iou and overlap > best_iou:
                best_iou = overlap
                best_gold_index = gold_index
        matches.append(
            {
                "pred_index": pred_index,
                "matched_gold_index": best_gold_index,
                "iou": best_iou,
                "matched": best_gold_index is not None,
            }
        )
        if best_gold_index is not None:
            unmatched_gold.remove(best_gold_index)
            tp += 1
    return tp, matches


def score_predictions(
    gold_samples: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    *,
    image_iou: float = 0.5,
    ignore_trigger: bool = False,
    comparison_preview: int = 5,
) -> dict[str, Any]:
    predictions_by_id = index_predictions_by_id(prediction_records)
    comparisons: list[dict[str, Any]] = []

    event_tp = 0
    event_pred_total = 0
    event_gold_total = 0

    text_tp = 0
    text_pred_total = 0
    text_gold_total = 0
    image_tp = 0
    image_pred_total = 0
    image_gold_total = 0

    gated_text_tp = 0
    gated_text_pred_total = 0
    gated_text_gold_total = 0
    gated_image_tp = 0
    gated_image_pred_total = 0
    gated_image_gold_total = 0
    valid_xmtl_samples = 0

    event_type_stats: dict[str, dict[str, int]] = {}
    event_type_stats_xmtl: dict[str, dict[str, int]] = {}
    role_stats: dict[str, dict[str, dict[str, int]]] = {}
    role_stats_xmtl: dict[str, dict[str, dict[str, int]]] = {}

    for sample in gold_samples:
        sample_id = str(sample.get("id") or "").strip()
        gold_event_type = resolve_gold_event_type(sample)
        prediction_record = predictions_by_id.get(sample_id)
        predicted_event_type = resolve_predicted_event_type(prediction_record)
        if gold_event_type:
            event_gold_total += 1
        if predicted_event_type:
            event_pred_total += 1
        if gold_event_type and predicted_event_type == gold_event_type:
            event_tp += 1

        gold_text = extract_gold_text_argument_tuples(sample, ignore_trigger=ignore_trigger)
        pred_text = extract_predicted_text_argument_tuples(prediction_record, ignore_trigger=ignore_trigger)
        gold_text_counter = Counter(gold_text)
        pred_text_counter = Counter(pred_text)
        text_sample_tp = sum((gold_text_counter & pred_text_counter).values())
        text_tp += text_sample_tp
        text_gold_total += sum(gold_text_counter.values())
        text_pred_total += sum(pred_text_counter.values())

        gold_image = extract_gold_image_arguments(sample)
        pred_image = extract_predicted_image_arguments(prediction_record)
        image_sample_tp, image_match_details = match_image_arguments(
            pred_image,
            gold_image,
            image_iou=image_iou,
        )
        image_tp += image_sample_tp
        image_gold_total += len(gold_image)
        image_pred_total += len(pred_image)

        event_stats = event_type_stats.setdefault(gold_event_type or "(missing)", {"tp": 0, "pred": 0, "gold": 0})
        event_stats["tp"] += text_sample_tp + image_sample_tp
        event_stats["pred"] += len(pred_text) + len(pred_image)
        event_stats["gold"] += len(gold_text) + len(gold_image)

        _accumulate_role_stats(
            role_stats,
            gold_event_type or "(missing)",
            gold_text,
            pred_text,
            text_sample_tp_counter=(gold_text_counter & pred_text_counter),
            gold_image=gold_image,
            pred_image=pred_image,
            image_matches=image_match_details,
        )

        xmtl_gate = bool(predicted_event_type and predicted_event_type == gold_event_type)
        if xmtl_gate:
            valid_xmtl_samples += 1
            gated_text_tp += text_sample_tp
            gated_text_gold_total += len(gold_text)
            gated_text_pred_total += len(pred_text)
            gated_image_tp += image_sample_tp
            gated_image_gold_total += len(gold_image)
            gated_image_pred_total += len(pred_image)

            gated_event_stats = event_type_stats_xmtl.setdefault(
                gold_event_type or "(missing)",
                {"tp": 0, "pred": 0, "gold": 0},
            )
            gated_event_stats["tp"] += text_sample_tp + image_sample_tp
            gated_event_stats["pred"] += len(pred_text) + len(pred_image)
            gated_event_stats["gold"] += len(gold_text) + len(gold_image)

            _accumulate_role_stats(
                role_stats_xmtl,
                gold_event_type or "(missing)",
                gold_text,
                pred_text,
                text_sample_tp_counter=(gold_text_counter & pred_text_counter),
                gold_image=gold_image,
                pred_image=pred_image,
                image_matches=image_match_details,
            )

        comparisons.append(
            {
                "sample_id": sample_id,
                "gold_event_type": gold_event_type,
                "predicted_event_type": predicted_event_type,
                "event_type_match": predicted_event_type == gold_event_type and bool(gold_event_type),
                "xmtl_gate": xmtl_gate,
                "gold_text_arguments": [list(item) for item in gold_text],
                "predicted_text_arguments": [list(item) for item in pred_text],
                "gold_image_arguments": gold_image,
                "predicted_image_arguments": pred_image,
                "matched_text_arguments": text_sample_tp,
                "matched_image_arguments": image_sample_tp,
            }
        )

    report = {
        "n_samples": len(gold_samples),
        "valid_xmtl_samples": valid_xmtl_samples,
        "event_extraction": _metric_block(
            tp=event_tp,
            pred=event_pred_total,
            gold=event_gold_total,
            accuracy=event_tp / len(gold_samples) if gold_samples else 0.0,
        ),
        "text_argument": _metric_block(tp=text_tp, pred=text_pred_total, gold=text_gold_total),
        "image_argument": _metric_block(tp=image_tp, pred=image_pred_total, gold=image_gold_total),
        "overall_argument_all_samples": _metric_block(
            tp=text_tp + image_tp,
            pred=text_pred_total + image_pred_total,
            gold=text_gold_total + image_gold_total,
        ),
        "overall_argument_xmtl_style": _metric_block(
            tp=gated_text_tp + gated_image_tp,
            pred=gated_text_pred_total + gated_image_pred_total,
            gold=gated_text_gold_total + gated_image_gold_total,
        ),
        "overall_argument_xmtl_style_text_only": _metric_block(
            tp=gated_text_tp,
            pred=gated_text_pred_total,
            gold=gated_text_gold_total,
        ),
        "overall_argument_xmtl_style_image_only": _metric_block(
            tp=gated_image_tp,
            pred=gated_image_pred_total,
            gold=gated_image_gold_total,
        ),
        "event_type_statistics": _materialize_metric_map(event_type_stats),
        "event_type_statistics_xmtl_style": _materialize_metric_map(event_type_stats_xmtl),
        "role_statistics_by_event_type": _materialize_nested_metric_map(role_stats),
        "role_statistics_by_event_type_xmtl_style": _materialize_nested_metric_map(role_stats_xmtl),
        "comparison_samples_preview": comparisons[: max(0, int(comparison_preview))],
    }
    return report


def _normalize_gold_image_argument(argument: Any) -> dict[str, Any] | None:
    if not isinstance(argument, Mapping):
        return None
    event_type = canonicalize_event_type(argument.get("event_type"))
    role = canonicalize_role(argument.get("role"))
    bbox = normalize_bbox(argument.get("bbox"))
    if not event_type or not role or bbox is None:
        return None
    return {
        "event_type": event_type,
        "role": role,
        "bbox": bbox,
    }


def _accumulate_role_stats(
    container: dict[str, dict[str, dict[str, int]]],
    event_type: str,
    gold_text: list[tuple[Any, ...]],
    pred_text: list[tuple[Any, ...]],
    *,
    text_sample_tp_counter: Counter[tuple[Any, ...]],
    gold_image: list[dict[str, Any]],
    pred_image: list[dict[str, Any]],
    image_matches: list[dict[str, Any]],
) -> None:
    event_bucket = container.setdefault(event_type, {})
    gold_text_counter = Counter(gold_text)
    pred_text_counter = Counter(pred_text)

    for item, count in gold_text_counter.items():
        role = str(item[-3] if len(item) == 6 else item[-3] if len(item) == 4 else "")
        role_bucket = event_bucket.setdefault(role, {"tp": 0, "pred": 0, "gold": 0})
        role_bucket["gold"] += count
    for item, count in pred_text_counter.items():
        role = str(item[-3] if len(item) == 6 else item[-3] if len(item) == 4 else "")
        role_bucket = event_bucket.setdefault(role, {"tp": 0, "pred": 0, "gold": 0})
        role_bucket["pred"] += count
    for item, count in text_sample_tp_counter.items():
        role = str(item[-3] if len(item) == 6 else item[-3] if len(item) == 4 else "")
        role_bucket = event_bucket.setdefault(role, {"tp": 0, "pred": 0, "gold": 0})
        role_bucket["tp"] += count

    for item in gold_image:
        role_bucket = event_bucket.setdefault(item["role"], {"tp": 0, "pred": 0, "gold": 0})
        role_bucket["gold"] += 1
    for item in pred_image:
        role_bucket = event_bucket.setdefault(item["role"], {"tp": 0, "pred": 0, "gold": 0})
        role_bucket["pred"] += 1
    for match in image_matches:
        if not match.get("matched"):
            continue
        pred_index = match["pred_index"]
        if not isinstance(pred_index, int) or pred_index < 0 or pred_index >= len(pred_image):
            continue
        role = pred_image[pred_index]["role"]
        role_bucket = event_bucket.setdefault(role, {"tp": 0, "pred": 0, "gold": 0})
        role_bucket["tp"] += 1


def _metric_block(*, tp: int, pred: int, gold: int, accuracy: float | None = None) -> dict[str, Any]:
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
    block: dict[str, Any] = {
        "tp": tp,
        "pred": pred,
        "gold": gold,
        "P": precision,
        "R": recall,
        "F1": f1,
    }
    if accuracy is not None:
        block["accuracy"] = accuracy
    return block


def _materialize_metric_map(raw: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, Any]]:
    return {
        key: _metric_block(tp=value["tp"], pred=value["pred"], gold=value["gold"])
        for key, value in raw.items()
    }


def _materialize_nested_metric_map(
    raw: Mapping[str, Mapping[str, Mapping[str, int]]]
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        event_type: {
            role: _metric_block(tp=counts["tp"], pred=counts["pred"], gold=counts["gold"])
            for role, counts in roles.items()
        }
        for event_type, roles in raw.items()
    }


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def main() -> None:
    args = parse_args()
    gold_samples = load_json_or_jsonl(args.gold)
    prediction_records = load_json_or_jsonl(args.pred)
    report = score_predictions(
        gold_samples,
        prediction_records,
        image_iou=float(args.image_iou),
        ignore_trigger=bool(args.ignore_trigger),
        comparison_preview=int(args.comparison_preview),
    )
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
