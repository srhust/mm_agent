from __future__ import annotations

import unittest

from scripts.score_m2e2_current import (
    bbox_iou,
    canonicalize_event_type,
    canonicalize_role_for_event,
    extract_predicted_text_argument_tuples,
    match_image_arguments,
    score_predictions,
)


class M2E2CurrentScorerTests(unittest.TestCase):
    def test_current_prediction_format_parsing_uses_nested_prediction_object(self) -> None:
        record = {
            "id": "sample-1",
            "prediction": {
                "event_type": "Justice:Arrest-Jail",
                "trigger": {"text": "custody", "start": 8, "end": 9},
                "text_arguments": [
                    {"role": "Person", "text": "man", "start": 6, "end": 7},
                ],
                "image_arguments": [],
            },
            "verified": True,
        }
        self.assertEqual(
            extract_predicted_text_argument_tuples(record, ignore_trigger=False),
            [("Justice:Arrest-Jail", 8, 9, "Person", 6, 7)],
        )

    def test_event_extraction_uses_event_type_only(self) -> None:
        gold = [
            {
                "id": "sample-1",
                "meta": {"crossmedia_event_type": "Justice:Arrest-Jail"},
                "text_event_mentions": [
                    {
                        "event_type": "Justice:Arrest-Jail",
                        "trigger": {"start": 8, "end": 9, "text": "custody"},
                        "arguments": [],
                    }
                ],
            }
        ]
        predictions = [
            {
                "id": "sample-1",
                "prediction": {
                    "event_type": "Justice:ArrestJail",
                    "trigger": {"text": "wrong", "start": 2, "end": 3},
                    "text_arguments": [],
                    "image_arguments": [],
                },
            }
        ]
        report = score_predictions(gold, predictions)
        self.assertEqual(report["event_extraction"]["tp"], 1)
        self.assertEqual(report["event_extraction"]["accuracy"], 1.0)

    def test_text_argument_exact_matching_respects_trigger_by_default(self) -> None:
        gold = [
            {
                "id": "sample-1",
                "meta": {"crossmedia_event_type": "Justice:Arrest-Jail"},
                "text_event_mentions": [
                    {
                        "event_type": "Justice:Arrest-Jail",
                        "trigger": {"start": 8, "end": 9, "text": "custody"},
                        "arguments": [{"role": "Person", "start": 6, "end": 7, "text": "man"}],
                    }
                ],
            }
        ]
        predictions = [
            {
                "id": "sample-1",
                "prediction": {
                    "event_type": "Justice:Arrest-Jail",
                    "trigger": {"text": "custody", "start": 8, "end": 9},
                    "text_arguments": [{"role": "Person", "text": "man", "start": 6, "end": 7}],
                    "image_arguments": [],
                },
            }
        ]
        report = score_predictions(gold, predictions)
        self.assertEqual(report["text_argument"]["tp"], 1)
        self.assertEqual(report["text_argument"]["F1"], 1.0)

    def test_image_iou_matching_uses_threshold(self) -> None:
        gold = [{"event_type": "Justice:Arrest-Jail", "role": "Agent", "bbox": [0.0, 0.0, 10.0, 10.0]}]
        pred = [{"event_type": "Justice:Arrest-Jail", "role": "Agent", "bbox": [1.0, 1.0, 11.0, 11.0]}]
        tp, matches = match_image_arguments(pred, gold, image_iou=0.5)
        self.assertEqual(tp, 1)
        self.assertTrue(matches[0]["matched"])
        self.assertGreater(bbox_iou(pred[0]["bbox"], gold[0]["bbox"]), 0.5)

    def test_xmtl_style_gating_requires_event_type_match(self) -> None:
        gold = [
            {
                "id": "sample-1",
                "meta": {"crossmedia_event_type": "Justice:Arrest-Jail"},
                "text_event_mentions": [
                    {
                        "event_type": "Justice:Arrest-Jail",
                        "trigger": {"start": 8, "end": 9, "text": "custody"},
                        "arguments": [{"role": "Person", "start": 6, "end": 7, "text": "man"}],
                    }
                ],
                "image_arguments_flat": [
                    {"event_type": "Justice:Arrest-Jail", "role": "Agent", "bbox": [0, 0, 10, 10]}
                ],
            }
        ]
        predictions = [
            {
                "id": "sample-1",
                "prediction": {
                    "event_type": "Conflict:Attack",
                    "trigger": {"text": "custody", "start": 8, "end": 9},
                    "text_arguments": [{"role": "Person", "text": "man", "start": 6, "end": 7}],
                    "image_arguments": [{"role": "Agent", "bbox": [0, 0, 10, 10]}],
                },
            }
        ]
        report = score_predictions(gold, predictions)
        self.assertEqual(report["overall_argument_all_samples"]["tp"], 0)
        self.assertEqual(report["valid_xmtl_samples"], 0)
        self.assertEqual(report["overall_argument_xmtl_style"]["pred"], 0)

    def test_missing_prediction_is_treated_as_empty(self) -> None:
        gold = [
            {
                "id": "sample-1",
                "meta": {"crossmedia_event_type": "Justice:Arrest-Jail"},
                "text_event_mentions": [],
            }
        ]
        report = score_predictions(gold, [])
        self.assertEqual(report["event_extraction"]["pred"], 0)
        self.assertEqual(report["event_extraction"]["gold"], 1)

    def test_canonicalize_event_type_handles_current_aliases(self) -> None:
        self.assertEqual(canonicalize_event_type("Justice:ArrestJail"), "Justice:Arrest-Jail")
        self.assertEqual(canonicalize_event_type("Contact:PhoneWrite"), "Contact:Phone-Write")
        self.assertEqual(canonicalize_event_type("Transaction:TransferMoney"), "Transaction:Transfer-Money")

    def test_contact_meet_entity_prediction_matches_participant_gold(self) -> None:
        gold = [
            {
                "id": "sample-1",
                "meta": {"crossmedia_event_type": "Contact:Meet"},
                "text_event_mentions": [
                    {
                        "event_type": "Contact:Meet",
                        "trigger": {"start": 3, "end": 4, "text": "met"},
                        "arguments": [{"role": "Participant", "start": 0, "end": 2, "text": "leaders"}],
                    }
                ],
                "image_arguments_flat": [
                    {"event_type": "Contact:Meet", "role": "Participant", "bbox": [0, 0, 10, 10]}
                ],
            }
        ]
        predictions = [
            {
                "id": "sample-1",
                "prediction": {
                    "event_type": "Contact:Meet",
                    "trigger": {"text": "met", "start": 3, "end": 4},
                    "text_arguments": [{"role": "Entity", "text": "leaders", "start": 0, "end": 2}],
                    "image_arguments": [{"role": "Entity", "bbox": [0, 0, 10, 10]}],
                },
            }
        ]

        report = score_predictions(gold, predictions)

        self.assertEqual(canonicalize_role_for_event("Contact:Meet", "Entity"), "Participant")
        self.assertEqual(report["text_argument"]["tp"], 1)
        self.assertEqual(report["image_argument"]["tp"], 1)
        self.assertEqual(report["role_alias_rescued_matches"], 2)
        self.assertEqual(report["role_alias_rescued_matches_by_event_type"]["Contact:Meet"], 2)
        self.assertIn(
            {"event_type": "Contact:Meet", "gold_role": "Participant", "pred_role": "Entity", "count": 2},
            report["raw_role_confusion_counts"],
        )
        self.assertNotIn(
            {"event_type": "Contact:Meet", "gold_role": "Participant", "pred_role": "Entity", "count": 2},
            report["normalized_role_confusion_counts"],
        )
        preview = report["comparison_samples_preview"][0]
        self.assertEqual(len(preview["alias_rescued_text_arguments"]), 1)
        self.assertEqual(len(preview["alias_rescued_image_arguments"]), 1)

    def test_contact_meet_participant_prediction_matches_entity_gold(self) -> None:
        gold = [
            {
                "id": "sample-1",
                "meta": {"crossmedia_event_type": "Contact:Meet"},
                "text_event_mentions": [
                    {
                        "event_type": "Contact:Meet",
                        "trigger": {"start": 3, "end": 4, "text": "met"},
                        "arguments": [{"role": "Entity", "start": 0, "end": 2, "text": "leaders"}],
                    }
                ],
                "image_arguments_flat": [
                    {"event_type": "Contact:Meet", "role": "Entity", "bbox": [0, 0, 10, 10]}
                ],
            }
        ]
        predictions = [
            {
                "id": "sample-1",
                "prediction": {
                    "event_type": "Contact:Meet",
                    "trigger": {"text": "met", "start": 3, "end": 4},
                    "text_arguments": [{"role": "Participant", "text": "leaders", "start": 0, "end": 2}],
                    "image_arguments": [{"role": "Participant", "bbox": [0, 0, 10, 10]}],
                },
            }
        ]

        report = score_predictions(gold, predictions)

        self.assertEqual(report["text_argument"]["tp"], 1)
        self.assertEqual(report["image_argument"]["tp"], 1)
        self.assertEqual(report["role_alias_rescued_matches"], 2)
        self.assertEqual(report["role_statistics_by_event_type"]["Contact:Meet"]["Participant"]["tp"], 2)

    def test_entity_participant_alias_is_not_global(self) -> None:
        gold = [
            {
                "id": "sample-1",
                "meta": {"crossmedia_event_type": "Contact:Phone-Write"},
                "text_event_mentions": [
                    {
                        "event_type": "Contact:Phone-Write",
                        "trigger": {"start": 3, "end": 4, "text": "called"},
                        "arguments": [{"role": "Entity", "start": 0, "end": 2, "text": "leaders"}],
                    }
                ],
                "image_arguments_flat": [
                    {"event_type": "Contact:Phone-Write", "role": "Entity", "bbox": [0, 0, 10, 10]}
                ],
            }
        ]
        predictions = [
            {
                "id": "sample-1",
                "prediction": {
                    "event_type": "Contact:Phone-Write",
                    "trigger": {"text": "called", "start": 3, "end": 4},
                    "text_arguments": [{"role": "Participant", "text": "leaders", "start": 0, "end": 2}],
                    "image_arguments": [{"role": "Participant", "bbox": [0, 0, 10, 10]}],
                },
            }
        ]

        report = score_predictions(gold, predictions)

        self.assertEqual(canonicalize_role_for_event("Contact:Phone-Write", "Participant"), "Participant")
        self.assertEqual(report["text_argument"]["tp"], 0)
        self.assertEqual(report["image_argument"]["tp"], 0)
        self.assertEqual(report["role_alias_rescued_matches"], 0)
        self.assertIn(
            {"event_type": "Contact:Phone-Write", "gold_role": "Entity", "pred_role": "Participant", "count": 2},
            report["raw_role_confusion_counts"],
        )
        self.assertIn(
            {"event_type": "Contact:Phone-Write", "gold_role": "Entity", "pred_role": "Participant", "count": 2},
            report["normalized_role_confusion_counts"],
        )

    def test_score_report_includes_role_diagnostics_for_empty_arguments(self) -> None:
        report = score_predictions(
            [{"id": "sample-1", "meta": {"crossmedia_event_type": "Contact:Meet"}, "text_event_mentions": []}],
            [],
        )

        self.assertIn("role_alias_rescued_matches", report)
        self.assertIn("role_alias_rescued_matches_by_event_type", report)
        self.assertIn("raw_role_confusion_counts", report)
        self.assertIn("normalized_role_confusion_counts", report)
        self.assertEqual(report["role_alias_rescued_matches"], 0)
        self.assertEqual(report["raw_role_confusion_counts"], [])


if __name__ == "__main__":
    unittest.main()
