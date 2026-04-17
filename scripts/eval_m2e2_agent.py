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
    m2e2_sample_to_agent_state,
)
from mm_event_agent.main import _initialize_rag_runtime
from scripts.run_m2e2_smoke import build_stage_trace_record, load_m2e2_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline M2E2-style evaluation with sanitized agent inputs.")
    parser.add_argument("--input", required=True, help="Path to a JSON or JSONL file containing M2E2-format samples.")
    parser.add_argument("--image-root", required=True, help="Root directory that contains sample image files.")
    parser.add_argument("--limit", type=int, default=10000, help="Number of samples to evaluate.")
    parser.add_argument("--output-dir", required=True, help="Directory where evaluation artifacts will be saved.")
    return parser.parse_args()


def evaluate_samples(samples: list[dict[str, Any]], image_root: str | Path) -> list[dict[str, Any]]:
    _initialize_rag_runtime()
    graph = build_graph()
    results: list[dict[str, Any]] = []
    for sample in samples:
        agent_input = m2e2_sample_to_agent_state(sample, image_root)
        final_state = graph.invoke(agent_input)
        gold = extract_m2e2_gold_record(sample)
        prediction = agent_output_to_m2e2_prediction(sample, final_state)
        results.append(
            {
                "sample_id": gold.get("sample_id"),
                "agent_input": agent_input,
                "gold": gold,
                "prediction": prediction,
                "verified": bool(final_state.get("verified")),
                "issues": list(final_state.get("issues", [])) if isinstance(final_state.get("issues"), list) else [],
                "trace": build_stage_trace_record(sample, agent_input, final_state),
            }
        )
    return results


def save_evaluation_outputs(results: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = target_dir / "predictions.jsonl"
    errors_path = target_dir / "errors.jsonl"
    summary_path = target_dir / "summary.json"
    trace_path = target_dir / "trace.jsonl"
    per_sample_metrics_path = target_dir / "per_sample_metrics.jsonl"

    with predictions_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(
                json.dumps(
                    {
                        **(item.get("prediction") if isinstance(item.get("prediction"), dict) else {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with errors_path.open("w", encoding="utf-8") as handle:
        for item in results:
            issues = item.get("issues")
            if not isinstance(issues, list) or not issues:
                continue
            handle.write(
                json.dumps(
                    {
                        "sample_id": item.get("sample_id"),
                        "issues": issues,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with trace_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item.get("trace", {}), ensure_ascii=False) + "\n")

    with per_sample_metrics_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(
                json.dumps(
                    {
                        "sample_id": item.get("sample_id"),
                        "verified": bool(item.get("verified")),
                        "issue_count": len(item.get("issues", [])) if isinstance(item.get("issues"), list) else 0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "count": len(results),
        "verified_count": sum(1 for item in results if item.get("verified")),
        "error_count": sum(
            1
            for item in results
            if isinstance(item.get("issues"), list) and bool(item.get("issues"))
        ),
        "files": {
            "predictions": str(predictions_path),
            "errors": str(errors_path),
            "summary": str(summary_path),
            "trace": str(trace_path),
            "per_sample_metrics": str(per_sample_metrics_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    samples = load_m2e2_samples(args.input)
    limited = samples[: max(0, int(args.limit))]
    results = evaluate_samples(limited, args.image_root)
    summary = save_evaluation_outputs(results, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
