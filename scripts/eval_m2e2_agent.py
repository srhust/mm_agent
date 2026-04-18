from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

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
from scripts.run_m2e2_smoke import (
    append_jsonl_record,
    build_prediction_record,
    build_stage_trace_record,
    build_progress_summary,
    initialize_output_paths,
    load_m2e2_samples,
    read_existing_processed_ids,
    safe_write_summary,
    write_result_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline M2E2-style evaluation with sanitized agent inputs.")
    parser.add_argument("--input", required=True, help="Path to a JSON or JSONL file containing M2E2-format samples.")
    parser.add_argument("--image-root", required=True, help="Root directory that contains sample image files.")
    parser.add_argument("--limit", type=int, default=10000, help="Number of samples to evaluate.")
    parser.add_argument("--output-dir", required=True, help="Directory where evaluation artifacts will be saved.")
    parser.add_argument("--resume", action="store_true", help="Skip sample ids already present in predictions/errors.")
    return parser.parse_args()


def evaluate_sample(
    sample: dict[str, Any],
    image_root: str | Path,
    *,
    graph: Any,
) -> dict[str, Any]:
    agent_input = m2e2_sample_to_agent_state(sample, image_root)
    final_state = graph.invoke(agent_input)
    gold = extract_m2e2_gold_record(sample)
    prediction = agent_output_to_m2e2_prediction(sample, final_state)
    return {
        "sample_id": gold.get("sample_id"),
        "agent_input": agent_input,
        "gold": gold,
        "prediction": prediction,
        "verified": bool(final_state.get("verified")),
        "issues": list(final_state.get("issues", [])) if isinstance(final_state.get("issues"), list) else [],
        "trace": build_stage_trace_record(sample, agent_input, final_state),
    }


def evaluate_samples(samples: list[dict[str, Any]], image_root: str | Path) -> list[dict[str, Any]]:
    _initialize_rag_runtime()
    graph = build_graph()
    return [evaluate_sample(sample, image_root, graph=graph) for sample in samples]


def save_evaluation_outputs(results: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    files = initialize_output_paths(output_dir)
    count = 0
    verified_count = 0
    error_count = 0
    for item in results:
        count += 1
        verified_count += 1 if item.get("verified") else 0
        error_count += 1 if isinstance(item.get("issues"), list) and bool(item.get("issues")) else 0
        write_result_record(
            output_dir,
            item,
            count=count,
            verified_count=verified_count,
            error_count=error_count,
            skipped_count=0,
        )
    summary = build_progress_summary(
        count=count,
        verified_count=verified_count,
        error_count=error_count,
        skipped_count=0,
        files=files,
    )
    safe_write_summary(files["summary"], summary)
    return summary


def run_incremental_evaluation(
    samples: list[dict[str, Any]],
    image_root: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    _initialize_rag_runtime()
    graph = build_graph()
    files = initialize_output_paths(output_dir)
    processed_ids = read_existing_processed_ids(output_dir) if resume else set()
    count = 0
    verified_count = 0
    error_count = 0
    skipped_count = 0

    if resume:
        metrics_path = Path(files["per_sample_metrics"])
        if metrics_path.exists():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                payload = line.strip()
                if not payload:
                    continue
                try:
                    record = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                count += 1
                verified_count += 1 if record.get("verified") else 0
                issue_count = record.get("issue_count")
                if isinstance(issue_count, int) and issue_count > 0:
                    error_count += 1

    safe_write_summary(
        files["summary"],
        build_progress_summary(
            count=count,
            verified_count=verified_count,
            error_count=error_count,
            skipped_count=skipped_count,
            files=files,
        ),
    )

    for sample in samples:
        sample_id = get_m2e2_sample_id(sample)
        if resume and sample_id and sample_id in processed_ids:
            skipped_count += 1
            safe_write_summary(
                files["summary"],
                build_progress_summary(
                    count=count,
                    verified_count=verified_count,
                    error_count=error_count,
                    skipped_count=skipped_count,
                    files=files,
                ),
            )
            continue
        result = evaluate_sample(sample, image_root, graph=graph)
        count += 1
        verified_count += 1 if result.get("verified") else 0
        error_count += 1 if isinstance(result.get("issues"), list) and bool(result.get("issues")) else 0
        write_result_record(
            output_dir,
            result,
            count=count,
            verified_count=verified_count,
            error_count=error_count,
            skipped_count=skipped_count,
        )
        if sample_id:
            processed_ids.add(sample_id)

    summary = build_progress_summary(
        count=count,
        verified_count=verified_count,
        error_count=error_count,
        skipped_count=skipped_count,
        files=files,
    )
    safe_write_summary(files["summary"], summary)
    return summary


def main() -> None:
    args = parse_args()
    samples = load_m2e2_samples(args.input)
    limited = samples[: max(0, int(args.limit))]
    summary = run_incremental_evaluation(
        limited,
        args.image_root,
        args.output_dir,
        resume=bool(args.resume),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
