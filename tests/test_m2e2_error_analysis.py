from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
import uuid

from scripts.analyze_m2e2_errors import (
    analyze_predictions,
    analyze_sample,
    write_analysis_outputs,
)


class M2E2ErrorAnalysisTests(unittest.TestCase):
    def test_event_type_mismatch_classification(self) -> None:
        gold = [
            {
                "id": "sample-1",
                "meta": {"crossmedia_event_type": "Conflict:Attack"},
                "text_event_mentions": [
                    {
                        "event_type": "Conflict:Attack",
                        "trigger": {"text": "exploded", "start": 2, "end": 3},
                        "arguments": [],
                    }
                ],
            }
        ]
        pred = [
            {
                "id": "sample-1",
                "prediction": {
                    "event_type": "Life:Die",
                    "trigger": {"text": "exploded", "start": 2, "end": 3},
                    "text_arguments": [],
                    "image_arguments": [],
                }
            }
        ]
        trace = [
            {
                "sample_id": "sample-1",
                "verifier_output": {"verified": True, "issues": []},
            }
        ]

        report = analyze_predictions(gold, pred, trace)

        self.assertEqual(report["error_summary"]["counts_per_category"]["event_type_mismatch"], 1)
        self.assertEqual(report["error_cases"][0]["likely_source_stages"], ["stage_a_event_type_selection", "verifier"])

    def test_text_boundary_overlong_detection(self) -> None:
        gold_sample = {
            "id": "sample-1",
            "meta": {"crossmedia_event_type": "Conflict:Attack"},
            "words": ["A", "bomb", "exploded", "in", "the", "busy", "market"],
            "text_event_mentions": [
                {
                    "event_type": "Conflict:Attack",
                    "trigger": {"text": "exploded", "start": 2, "end": 3},
                    "arguments": [{"role": "Place", "text": "market", "start": 6, "end": 7}],
                }
            ],
        }
        prediction = {
            "id": "sample-1",
            "prediction": {
                "event_type": "Conflict:Attack",
                "trigger": {"text": "exploded", "start": 2, "end": 3},
                "text_arguments": [{"role": "Place", "text": "the busy market", "start": 4, "end": 7}],
                "image_arguments": [],
            },
        }
        trace = {
            "sample_id": "sample-1",
            "stage_b_output": {"text_arguments": [{"role": "Place", "text": "the busy market"}]},
            "verifier_output": {"verified": True, "issues": []},
        }

        case = analyze_sample(gold_sample, prediction, trace, None, image_iou=0.5)

        self.assertIn("text_arg_boundary_overlong", case["triggered_error_categories"])
        self.assertIn("stage_b_text_boundary", case["likely_source_stages"])

    def test_image_same_role_shortfall_detection(self) -> None:
        gold_sample = {
            "id": "sample-1",
            "meta": {"crossmedia_event_type": "Justice:Arrest-Jail"},
            "image_arguments_flat": [
                {"event_type": "Justice:Arrest-Jail", "role": "Person", "bbox": [0, 0, 10, 10]},
                {"event_type": "Justice:Arrest-Jail", "role": "Person", "bbox": [20, 20, 30, 30]},
            ],
        }
        prediction = {
            "id": "sample-1",
            "prediction": {
                "event_type": "Justice:Arrest-Jail",
                "trigger": None,
                "text_arguments": [],
                "image_arguments": [{"role": "Person", "bbox": [0, 0, 10, 10]}],
            },
        }
        trace = {
            "sample_id": "sample-1",
            "stage_c_output": {"image_arguments": [{"role": "Person", "label": "suspect one"}]},
            "grounding_results": [{"role": "Person", "label": "suspect one", "bbox": [0, 0, 10, 10], "grounding_status": "grounded"}],
            "verifier_output": {"verified": True, "issues": []},
        }

        case = analyze_sample(gold_sample, prediction, trace, None, image_iou=0.5)

        self.assertIn("image_arg_same_role_instance_shortfall", case["triggered_error_categories"])
        self.assertIn("stage_c_missing_candidate", case["triggered_error_categories"])

    def test_stage_attribution_from_trace_prefers_grounding_failure_when_stage_c_candidate_exists(self) -> None:
        gold_sample = {
            "id": "sample-1",
            "meta": {"crossmedia_event_type": "Conflict:Attack"},
            "image_arguments_flat": [
                {"event_type": "Conflict:Attack", "role": "Place", "bbox": [0, 0, 10, 10]},
            ],
        }
        prediction = {
            "id": "sample-1",
            "prediction": {
                "event_type": "Conflict:Attack",
                "trigger": None,
                "text_arguments": [],
                "image_arguments": [],
            },
        }
        trace = {
            "sample_id": "sample-1",
            "stage_c_output": {"image_arguments": [{"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}]},
            "grounding_results": [{"role": "Place", "label": "market area", "bbox": None, "grounding_status": "failed"}],
            "verifier_output": {"verified": False, "issues": ["missing image support"]},
        }

        case = analyze_sample(gold_sample, prediction, trace, None, image_iou=0.5)

        self.assertIn("grounding_failed_after_stage_c_candidate", case["triggered_error_categories"])
        self.assertIn("grounding_failed_after_stage_c_candidate", case["likely_source_stages"])

    def test_weak_place_hallucination_classification_when_present(self) -> None:
        gold_sample = {
            "id": "sample-1",
            "meta": {"crossmedia_event_type": "Conflict:Attack"},
            "image_arguments_flat": [],
        }
        prediction = {
            "id": "sample-1",
            "prediction": {
                "event_type": "Conflict:Attack",
                "trigger": None,
                "text_arguments": [],
                "image_arguments": [{"role": "Place", "bbox": [0, 0, 10, 10]}],
            },
        }
        trace = {
            "sample_id": "sample-1",
            "stage_c_output": {"image_arguments": [{"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}]},
            "grounding_results": [{"role": "Place", "label": "market area", "bbox": [0, 0, 10, 10], "grounding_status": "grounded"}],
            "verifier_output": {"verified": True, "issues": []},
        }

        case = analyze_sample(gold_sample, prediction, trace, None, image_iou=0.5)

        self.assertIn("image_arg_generic_weak_place", case["triggered_error_categories"])
        self.assertIn("stage_c_weak_place_proposal", case["likely_source_stages"])

    def test_write_analysis_outputs_creates_expected_files(self) -> None:
        output_dir = Path(f"test_error_analysis_{uuid.uuid4().hex}")
        shutil.rmtree(output_dir, ignore_errors=True)
        try:
            written = write_analysis_outputs(
                output_dir,
                error_summary={"counts_per_category": {"event_type_mismatch": 1}},
                error_cases=[{"id": "sample-1", "triggered_error_categories": ["event_type_mismatch"]}],
            )
            self.assertTrue((output_dir / "error_summary.json").exists())
            self.assertTrue((output_dir / "error_cases.jsonl").exists())
            self.assertEqual(json.loads((output_dir / "error_summary.json").read_text(encoding="utf-8"))["counts_per_category"]["event_type_mismatch"], 1)
            self.assertEqual(written["error_summary"], str(output_dir / "error_summary.json"))
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
