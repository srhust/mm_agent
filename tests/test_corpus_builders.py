from __future__ import annotations

import unittest

from mm_event_agent.rag.normalizers import Ace2005Normalizer, MavenArgNormalizer, SwigNormalizer
from mm_event_agent.rag.ontology_mapper import OntologyMapper
from scripts.build_ace2005_corpus import flatten_ace_record
from scripts.build_maven_arg_corpus import flatten_maven_document
from scripts.build_swig_corpus import flatten_swig_record


class CorpusBuilderFlatteningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = OntologyMapper()
        self.ace_normalizer = Ace2005Normalizer(self.mapper)
        self.maven_normalizer = MavenArgNormalizer(self.mapper)
        self.swig_normalizer = SwigNormalizer(self.mapper)

    def test_maven_document_level_record_flattens_to_event_level_sample(self) -> None:
        raw_document = {
            "id": "doc-1",
            "document": "A civilian was killed downtown.",
            "entities": [
                {
                    "id": "ent-place",
                    "mention": [{"mention": "downtown", "offset": [22, 30]}],
                }
            ],
            "events": [
                {
                    "id": "event-1",
                    "type": "Life:Die",
                    "mention": [{"id": "mention-1", "trigger_word": "killed", "offset": [15, 21]}],
                    "argument": {
                        "Victim": [{"content": "civilian", "offset": [2, 10]}],
                        "Place": [{"entity_id": "ent-place"}],
                    },
                }
            ],
        }

        flattened = flatten_maven_document(raw_document)

        self.assertEqual(len(flattened), 1)
        self.assertEqual(flattened[0]["id"], "mention-1")
        self.assertEqual(flattened[0]["trigger"]["text"], "killed")
        self.assertEqual(len(flattened[0]["arguments"]), 2)
        normalized = self.maven_normalizer.normalize(flattened[0])
        self.assertIsNotNone(normalized)

    def test_unresolved_maven_entity_argument_is_dropped_but_event_is_kept(self) -> None:
        raw_document = {
            "id": "doc-2",
            "document": "Protesters met in the square.",
            "entities": [],
            "events": [
                {
                    "id": "event-2",
                    "type": "Contact:Meet",
                    "mention": [{"id": "mention-2", "trigger_word": "met", "offset": [11, 14]}],
                    "argument": {
                        "Participant": [{"content": "Protesters", "offset": [0, 10]}],
                        "Place": [{"entity_id": "missing-entity"}],
                    },
                }
            ],
        }

        flattened = flatten_maven_document(raw_document)

        self.assertEqual(len(flattened), 1)
        self.assertEqual(len(flattened[0]["arguments"]), 1)
        normalized = self.maven_normalizer.normalize(flattened[0])
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(len(normalized["text_arguments"]), 1)

    def test_ace_line_level_record_flattens_event_list_and_generates_doc_id(self) -> None:
        raw_record = {
            "text": "The bomb exploded near the market.",
            "event": [
                {
                    "type": "Conflict:Attack",
                    "text": "exploded",
                    "args": [
                        {"type": "Attacker", "text": "bombers"},
                        {"type": "Place", "text": "market"},
                    ],
                }
            ],
        }

        flattened = flatten_ace_record(raw_record, split_name="train", line_number=7)

        self.assertEqual(len(flattened), 1)
        self.assertEqual(flattened[0]["doc_id"], "train::000007")
        self.assertEqual(flattened[0]["trigger"]["text"], "exploded")
        normalized = self.ace_normalizer.normalize(flattened[0])
        self.assertIsNotNone(normalized)

    def test_swig_keyed_dict_record_flattens_to_intermediate_sample(self) -> None:
        flattened = flatten_swig_record(
            "attacking_1.jpg",
            {
                "verb": "attacking",
                "frames": [
                    {"agent": "soldier", "weapon": "rifle", "target": "truck"},
                    {"agent": "soldier"},
                ],
                "width": 512,
                "height": 512,
            },
            mapper=self.mapper,
        )

        self.assertEqual(flattened["image_id"], "attacking_1.jpg")
        self.assertEqual(flattened["path"], "attacking_1.jpg")
        self.assertEqual(flattened["arguments"][0]["role"], "agent")
        self.assertIn("Conflict:Attack", flattened["image_desc"])

    def test_missing_image_file_does_not_crash_swig_normalization(self) -> None:
        flattened = flatten_swig_record(
            "meeting_1.jpg",
            {
                "verb": "meeting",
                "frames": [{"agent": "delegate", "place": "hall"}],
            },
            mapper=self.mapper,
        )

        normalized = self.swig_normalizer.normalize(flattened, images_root="E:/does-not-exist")

        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertTrue(normalized.image_manifest["path"].endswith("meeting_1.jpg"))


if __name__ == "__main__":
    unittest.main()
