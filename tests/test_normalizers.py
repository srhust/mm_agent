from __future__ import annotations

import unittest

from mm_event_agent.rag.normalizers import (
    Ace2005Normalizer,
    MavenArgNormalizer,
    SwigNormalizer,
)
from mm_event_agent.rag.ontology_mapper import OntologyMapper


class NormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        mapper = OntologyMapper()
        self.ace = Ace2005Normalizer(mapper)
        self.maven = MavenArgNormalizer(mapper)
        self.swig = SwigNormalizer(mapper)

    def test_ace_like_sample_normalizes_to_text_contract(self) -> None:
        record = self.ace.normalize(
            {
                "id": "ace-1",
                "doc_id": "doc-1",
                "event_type": "Conflict:Attack",
                "text": "A bomb exploded in the market.",
                "trigger": {"text": "exploded", "span": {"start": 7, "end": 16}},
                "arguments": [
                    {"role": "Attacker", "text": "bombers"},
                    {"role": "Place", "text": "market", "span": {"start": 24, "end": 30}},
                ],
            }
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["source_dataset"], "ACE2005")
        self.assertEqual(record["event_type"], "Conflict:Attack")
        self.assertEqual(record["trigger"]["text"], "exploded")
        self.assertEqual(record["text_arguments"][0]["role"], "Attacker")
        self.assertIn("Conflict:Attack", record["retrieval_text"])

    def test_maven_like_sample_normalizes_to_text_contract(self) -> None:
        record = self.maven.normalize(
            {
                "id": "maven-1",
                "doc_id": "doc-2",
                "event_type": "Life:Die",
                "sentence": "A civilian was killed downtown.",
                "trigger_text": "killed",
                "arguments": [
                    {"role": "Victim", "text": "civilian"},
                    {"role": "Place", "text": "downtown"},
                ],
            }
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["source_dataset"], "MAVEN-ARG")
        self.assertEqual(record["event_type"], "Life:Die")
        self.assertEqual(record["trigger"]["text"], "killed")
        self.assertEqual(len(record["text_arguments"]), 2)

    def test_swig_like_sample_normalizes_both_text_and_image_records(self) -> None:
        record = self.swig.normalize(
            {
                "image_id": "img-1",
                "verb": "attacking",
                "frame": "attack_frame",
                "description": "Smoke and people running near damaged cars.",
                "arguments": [
                    {"role": "weapon", "label": "rifle"},
                    {"role": "target", "label": "car"},
                ],
                "file_name": "images/sample.jpg",
            },
            images_root="e:/mm_agent",
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.text_example["event_type"], "Conflict:Attack")
        self.assertEqual(record.text_example["image_arguments"][0]["role"], "Instrument")
        self.assertTrue(record.image_manifest["path"].endswith("images\\sample.jpg"))

    def test_unmapped_event_type_is_skipped_cleanly(self) -> None:
        record = self.ace.normalize(
            {
                "id": "ace-x",
                "doc_id": "doc-x",
                "event_type": "Unknown:Event",
                "text": "Nothing happened.",
                "trigger_text": "happened",
            }
        )

        self.assertIsNone(record)
        self.assertEqual(self.ace.last_skip_reason, "unmapped_event_type:unknown:event")

    def test_unmapped_role_is_dropped_but_sample_is_kept(self) -> None:
        record = self.swig.normalize(
            {
                "image_id": "img-2",
                "verb": "meeting",
                "description": "Two people sit around a table.",
                "arguments": [
                    {"role": "banana", "label": "banana"},
                    {"role": "person", "label": "manager"},
                ],
                "file_name": "meeting.jpg",
            }
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(len(record.text_example["image_arguments"]), 1)
        self.assertEqual(record.text_example["image_arguments"][0]["role"], "Participant")


if __name__ == "__main__":
    unittest.main()
