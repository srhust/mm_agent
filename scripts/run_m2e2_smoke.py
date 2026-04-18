from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_project_dotenv() -> bool:
    dotenv_path = PROJECT_ROOT / ".env"
    if not dotenv_path.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(dotenv_path=dotenv_path, override=False))


_DOTENV_LOADED = _load_project_dotenv()

from mm_event_agent.graph import build_graph
from mm_event_agent.m2e2_adapter import (
    agent_output_to_m2e2_prediction,
    extract_m2e2_gold_record,
    get_m2e2_sample_id,
    m2e2_sample_to_agent_state,
)
from mm_event_agent.main import _initialize_rag_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one M2E2-format sample through the current agent graph.")
    parser.add_argument("--input", required=True, help="Path to a JSON or JSONL file containing M2E2-format samples.")
    parser.add_argument("--image-root", required=True, help="Root directory that contains sample image files.")
    parser.add_argument("--sample-id", help="Optional sample id to select from the input file.")
    parser.add_argument("--output-dir", help="Optional directory for saved smoke artifacts.")
    return parser.parse_args()


def load_m2e2_samples(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".jsonl":
        samples: list[dict[str, Any]] = []
        for line in text.splitlines():
            payload = line.strip()
            if not payload:
                continue
            data = json.loads(payload)
            if isinstance(data, dict):
                samples.append(data)
        return samples

    data = json.loads(text)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("samples", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    raise ValueError(f"Unsupported input payload in {input_path}")


def select_sample(samples: Iterable[dict[str, Any]], sample_id: str | None) -> dict[str, Any]:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("No samples found in input file")
    if not sample_id:
        return sample_list[0]
    for sample in sample_list:
        if get_m2e2_sample_id(sample) == str(sample_id):
            return sample
    raise ValueError(f"Sample id not found: {sample_id}")


def summarize_similar_events(similar_events: Any) -> dict[str, int]:
    if not isinstance(similar_events, Mapping):
        return {
            "text_event_examples": 0,
            "image_semantic_examples": 0,
            "bridge_examples": 0,
        }
    return {
        "text_event_examples": len(similar_events.get("text_event_examples", []))
        if isinstance(similar_events.get("text_event_examples"), list)
        else 0,
        "image_semantic_examples": len(similar_events.get("image_semantic_examples", []))
        if isinstance(similar_events.get("image_semantic_examples"), list)
        else 0,
        "bridge_examples": len(similar_events.get("bridge_examples", []))
        if isinstance(similar_events.get("bridge_examples"), list)
        else 0,
    }


def build_stage_trace_record(sample: Mapping[str, Any], agent_input: Mapping[str, Any], final_state: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = get_m2e2_sample_id(sample)
    prompt_trace = final_state.get("prompt_trace")
    normalized_prompt_trace: list[dict[str, Any]] = []
    if isinstance(prompt_trace, list):
        for item in prompt_trace:
            if not isinstance(item, Mapping):
                continue
            record = dict(item)
            record["sample_id"] = sample_id
            normalized_prompt_trace.append(record)
    stage_outputs = final_state.get("stage_outputs")
    if not isinstance(stage_outputs, Mapping):
        stage_outputs = {}
    return {
        "sample_id": sample_id,
        "agent_input": dict(agent_input),
        "prompt_trace": normalized_prompt_trace,
        "perception_output": {
            "image_desc": str(final_state.get("image_desc") or ""),
            "perception_summary": str(final_state.get("perception_summary") or ""),
        },
        "similar_events_summary": stage_outputs.get("similar_events_summary", summarize_similar_events(final_state.get("similar_events"))),
        "evidence_summary": stage_outputs.get(
            "evidence_summary",
            {
                "count": len(final_state.get("evidence", [])) if isinstance(final_state.get("evidence"), list) else 0,
                "items": list(final_state.get("evidence", [])) if isinstance(final_state.get("evidence"), list) else [],
            },
        ),
        "fusion_context_summary": stage_outputs.get("fusion_context_summary", final_state.get("fusion_context", {})),
        "stage_a_output": stage_outputs.get("stage_a_output", {}),
        "stage_b_output": stage_outputs.get("stage_b_output", {}),
        "stage_c_output": stage_outputs.get("stage_c_output", {}),
        "grounding_requests": list(stage_outputs.get("grounding_requests", []))
        if isinstance(stage_outputs.get("grounding_requests"), list)
        else [],
        "grounding_results": list(final_state.get("grounding_results", [])) if isinstance(final_state.get("grounding_results"), list) else [],
        "verifier_output": stage_outputs.get(
            "verifier_output",
            {
                "verified": bool(final_state.get("verified")),
                "issues": list(final_state.get("issues", [])) if isinstance(final_state.get("issues"), list) else [],
                "verifier_reason": str(final_state.get("verifier_reason") or ""),
                "verifier_confidence": float(final_state.get("verifier_confidence") or 0.0),
                "verifier_diagnostics": list(final_state.get("verifier_diagnostics", []))
                if isinstance(final_state.get("verifier_diagnostics"), list)
                else [],
            },
        ),
        "repair_history": list(final_state.get("repair_history", [])) if isinstance(final_state.get("repair_history"), list) else [],
    }


def append_jsonl_record(path: str | Path, record: Mapping[str, Any]) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        handle.flush()


def safe_write_summary(path: str | Path, summary_obj: Mapping[str, Any]) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(dict(summary_obj), ensure_ascii=False, indent=2), encoding="utf-8")


def read_existing_processed_ids(output_dir: str | Path) -> set[str]:
    target_dir = Path(output_dir)
    processed: set[str] = set()
    for filename, id_key in (("predictions.jsonl", "id"), ("errors.jsonl", "sample_id")):
        path = target_dir / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = line.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, Mapping):
                continue
            value = record.get(id_key)
            if value is not None and str(value).strip():
                processed.add(str(value).strip())
    return processed


def build_prediction_record(sample: Mapping[str, Any], final_state: Mapping[str, Any]) -> dict[str, Any]:
    return agent_output_to_m2e2_prediction(sample, final_state)


def build_eval_result_record(
    sample: Mapping[str, Any],
    image_root: str | Path,
    final_state: Mapping[str, Any],
) -> dict[str, Any]:
    agent_input = m2e2_sample_to_agent_state(sample, image_root)
    gold = extract_m2e2_gold_record(sample)
    prediction = build_prediction_record(sample, final_state)
    return {
        "sample_id": gold.get("sample_id"),
        "agent_input": agent_input,
        "gold": gold,
        "prediction": prediction,
        "verified": bool(final_state.get("verified")),
        "issues": list(final_state.get("issues", [])) if isinstance(final_state.get("issues"), list) else [],
        "trace": build_stage_trace_record(sample, agent_input, final_state),
    }


def initialize_output_paths(output_dir: str | Path) -> dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "predictions": str(target_dir / "predictions.jsonl"),
        "trace": str(target_dir / "trace.jsonl"),
        "errors": str(target_dir / "errors.jsonl"),
        "summary": str(target_dir / "summary.json"),
        "per_sample_metrics": str(target_dir / "per_sample_metrics.jsonl"),
    }
    for key in ("predictions", "trace", "errors", "per_sample_metrics"):
        path = Path(files[key])
        with path.open("a", encoding="utf-8"):
            pass
    return files


def build_progress_summary(
    *,
    count: int,
    verified_count: int,
    error_count: int,
    skipped_count: int,
    files: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "count": int(count),
        "verified_count": int(verified_count),
        "error_count": int(error_count),
        "skipped_count": int(skipped_count),
        "files": dict(files),
    }


def write_result_record(
    output_dir: str | Path,
    result: Mapping[str, Any],
    *,
    count: int,
    verified_count: int,
    error_count: int,
    skipped_count: int = 0,
) -> dict[str, Any]:
    files = initialize_output_paths(output_dir)
    prediction = result.get("prediction") if isinstance(result.get("prediction"), Mapping) else {}
    append_jsonl_record(files["predictions"], prediction if isinstance(prediction, Mapping) else {})
    append_jsonl_record(files["trace"], result.get("trace") if isinstance(result.get("trace"), Mapping) else {})
    append_jsonl_record(
        files["per_sample_metrics"],
        {
            "sample_id": result.get("sample_id"),
            "verified": bool(result.get("verified")),
            "issue_count": len(result.get("issues", [])) if isinstance(result.get("issues"), list) else 0,
        },
    )

    issues = result.get("issues")
    if isinstance(issues, list) and issues:
        append_jsonl_record(
            files["errors"],
            {
                "sample_id": result.get("sample_id"),
                "issues": issues,
            },
        )

    summary = build_progress_summary(
        count=count,
        verified_count=verified_count,
        error_count=error_count,
        skipped_count=skipped_count,
        files=files,
    )
    safe_write_summary(files["summary"], summary)
    return summary


def save_smoke_artifacts(
    output_dir: str | Path,
    *,
    sample: Mapping[str, Any],
    agent_input: Mapping[str, Any],
    prediction: Mapping[str, Any],
    final_state: Mapping[str, Any],
) -> dict[str, str]:
    result = {
        "sample_id": get_m2e2_sample_id(sample) or "sample",
        "agent_input": dict(agent_input),
        "prediction": dict(prediction),
        "verified": bool(final_state.get("verified")),
        "issues": list(final_state.get("issues", [])) if isinstance(final_state.get("issues"), list) else [],
        "trace": build_stage_trace_record(sample, agent_input, final_state),
    }
    summary = write_result_record(
        output_dir,
        result,
        count=1,
        verified_count=1 if result["verified"] else 0,
        error_count=1 if result["issues"] else 0,
        skipped_count=0,
    )
    return dict(summary["files"])

def main() -> None:
    args = parse_args()
    samples = load_m2e2_samples(args.input)
    sample = select_sample(samples, args.sample_id)
    agent_input = m2e2_sample_to_agent_state(sample, args.image_root)

    raw_image = agent_input.get("raw_image")
    if isinstance(raw_image, str) and raw_image and not Path(raw_image).exists():
        raise FileNotFoundError(f"Image file not found: {raw_image}")

    # Respect the current runtime configuration: persistent, demo, or no-RAG.
    _initialize_rag_runtime()
    graph = build_graph()
    final_state = graph.invoke(agent_input)
    prediction = agent_output_to_m2e2_prediction(sample, final_state)
    gold = extract_m2e2_gold_record(sample)
    stage_trace = build_stage_trace_record(sample, agent_input, final_state)

    print(f"sample_id: {get_m2e2_sample_id(sample) or '(missing id)'}")
    print("gold:")
    print(json.dumps(gold, ensure_ascii=False, indent=2))
    print("predicted_event:")
    print(json.dumps(prediction, ensure_ascii=False, indent=2))
    print("verifier:")
    print(
        json.dumps(
            {
                "verified": bool(final_state.get("verified")),
                "issues": final_state.get("issues", []),
                "verifier_reason": final_state.get("verifier_reason", ""),
                "verifier_confidence": final_state.get("verifier_confidence", 0.0),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("similar_events_summary:")
    print(json.dumps(summarize_similar_events(final_state.get("similar_events")), ensure_ascii=False, indent=2))
    print("stage_outputs:")
    print(json.dumps(stage_trace, ensure_ascii=False, indent=2))
    if args.output_dir:
        saved = save_smoke_artifacts(
            args.output_dir,
            sample=sample,
            agent_input=agent_input,
            prediction=prediction,
            final_state=final_state,
        )
        print("saved_files:")
        print(json.dumps(saved, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
