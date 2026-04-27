from __future__ import annotations

import unittest

from scripts.analyze_m2e2_error_breakdown import analyze_error_breakdown, classify_box_quality


class M2E2ErrorBreakdownTests(unittest.TestCase):
    def test_contact_meet_entity_participant_uses_scorer_canonicalization(self) -> None:
        gold = [
            {
                "id": "sample-1",
                "meta": {"crossmedia_event_type": "Contact:Meet"},
                "text_event_mentions": [
                    {
                        "event_type": "Contact:Meet",
                        "trigger": {"start": 3, "end": 4},
                        "arguments": [{"role": "Participant", "start": 0, "end": 2}],
                    }
                ],
                "image_arguments_flat": [
                    {"event_type": "Contact:Meet", "role": "Participant", "bbox": [0, 0, 10, 10]}
                ],
            }
        ]
        pred = [
            {
                "id": "sample-1",
                "prediction": {
                    "event_type": "Contact:Meet",
                    "trigger": {"start": 3, "end": 4},
                    "text_arguments": [{"role": "Entity", "start": 0, "end": 2}],
                    "image_arguments": [{"role": "Entity", "bbox": [0, 0, 10, 10]}],
                },
            }
        ]

        report = analyze_error_breakdown(gold, pred)

        self.assertEqual(report["summary"]["text"]["text.true_positive"], 1)
        self.assertEqual(report["summary"]["image"]["image.true_positive"], 1)
        self.assertEqual(report["summary"]["text"]["text.fp_wrong_role_same_span"], 0)
        self.assertEqual(report["summary"]["image"]["image.fp_wrong_role_correct_box"], 0)

    def test_text_breakdown_splits_wrong_role_boundary_and_schema_errors(self) -> None:
        gold = [
            {
                "id": "sample-1",
                "meta": {"crossmedia_event_type": "Conflict:Attack"},
                "text_event_mentions": [
                    {
                        "event_type": "Conflict:Attack",
                        "trigger": {"start": 5, "end": 6},
                        "arguments": [
                            {"role": "Target", "start": 0, "end": 2},
                            {"role": "Attacker", "start": 7, "end": 9},
                            {"role": "Place", "start": 12, "end": 14},
                        ],
                    }
                ],
            }
        ]
        pred = [
            {
                "id": "sample-1",
                "prediction": {
                    "event_type": "Conflict:Attack",
                    "trigger": {"start": 5, "end": 6},
                    "text_arguments": [
                        {"role": "Attacker", "start": 0, "end": 2},
                        {"role": "Attacker", "start": 7, "end": 10},
                        {"role": "BogusRole", "start": 20, "end": 21},
                    ],
                    "image_arguments": [],
                },
            }
        ]

        report = analyze_error_breakdown(gold, pred)
        text = report["summary"]["text"]

        self.assertEqual(text["text.fp_wrong_role_same_span"], 1)
        self.assertEqual(text["text.fn_wrong_role_same_span"], 1)
        self.assertEqual(text["text.fp_boundary_error_same_role"], 1)
        self.assertEqual(text["text.fn_boundary_error_same_role"], 1)
        self.assertEqual(text["text.fp_role_outside_schema"], 1)
        self.assertEqual(text["text.pred_role_illegal_under_pred_event"], 1)
        self.assertEqual(text["text.fn_missing_argument"], 1)
        self.assertIn("text.fp_wrong_role_same_span", report["example_samples"])

    def test_image_breakdown_splits_wrong_role_and_box_quality(self) -> None:
        gold = [
            {
                "id": "sample-1",
                "meta": {"crossmedia_event_type": "Conflict:Attack"},
                "text_event_mentions": [],
                "image_arguments_flat": [
                    {"event_type": "Conflict:Attack", "role": "Target", "bbox": [0, 0, 10, 10]},
                    {"event_type": "Conflict:Attack", "role": "Instrument", "bbox": [20, 20, 30, 30]},
                    {"event_type": "Conflict:Attack", "role": "Place", "bbox": [40, 40, 50, 50]},
                ],
            }
        ]
        pred = [
            {
                "id": "sample-1",
                "prediction": {
                    "event_type": "Conflict:Attack",
                    "trigger": None,
                    "text_arguments": [],
                    "image_arguments": [
                        {"role": "Attacker", "bbox": [0, 0, 10, 10]},
                        {"role": "Instrument", "bbox": [18, 18, 34, 34]},
                        {"role": "Place", "bbox": [80, 80, 90, 90]},
                        {"role": "Target", "bbox": None, "grounding_status": "unresolved"},
                    ],
                },
            }
        ]

        report = analyze_error_breakdown(gold, pred, iou_threshold=0.5)
        image = report["summary"]["image"]

        self.assertEqual(image["image.fp_wrong_role_correct_box"], 1)
        self.assertEqual(image["image.fn_wrong_role_correct_box"], 1)
        self.assertEqual(image["image.fp_box_not_precise_enough_oversized"], 1)
        self.assertEqual(image["image.fn_box_not_precise_enough_oversized"], 1)
        self.assertEqual(image["image.fp_no_overlap_same_role"], 1)
        self.assertEqual(image["image.fn_no_overlap_same_role"], 1)
        self.assertEqual(image["image.fp_spurious_argument"], 1)
        self.assertIn("image.fp_spurious_argument", report["example_samples"])

    def test_box_quality_classifier(self) -> None:
        self.assertEqual(
            classify_box_quality([0, 0, 10, 10], [0, 0, 10, 10], iou_threshold=0.5, iou_low=0.2, cover_high=0.8, purity_low=0.6),
            "matched",
        )
        self.assertEqual(
            classify_box_quality([0, 0, 20, 20], [2, 2, 8, 8], iou_threshold=0.5, iou_low=0.05, cover_high=0.8, purity_low=0.6),
            "box_not_precise_enough_oversized",
        )
        self.assertEqual(
            classify_box_quality([2, 2, 8, 8], [0, 0, 20, 20], iou_threshold=0.5, iou_low=0.05, cover_high=0.8, purity_low=0.6),
            "box_not_precise_enough_undersized",
        )
        self.assertEqual(
            classify_box_quality([0, 0, 10, 10], [30, 30, 40, 40], iou_threshold=0.5, iou_low=0.2, cover_high=0.8, purity_low=0.6),
            "no_overlap",
        )


if __name__ == "__main__":
    unittest.main()
