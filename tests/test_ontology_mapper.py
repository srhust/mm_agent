from __future__ import annotations

import unittest

from mm_event_agent.rag.ontology_mapper import OntologyMapper


class OntologyMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = OntologyMapper()

    def test_ace2005_mapping_returns_canonical_event_and_role(self) -> None:
        event_type = self.mapper.map_event_type("ace2005", "Conflict:Attack")
        role = self.mapper.map_role("ace2005", "Conflict:Attack", "Attacker")

        self.assertEqual(event_type, "Conflict:Attack")
        self.assertEqual(role, "Attacker")

    def test_ace2005_lowercase_raw_labels_map_to_canonical_types(self) -> None:
        event_type = self.mapper.map_event_type("ace2005", "phone write")
        role = self.mapper.map_role("ace2005", "meet", "entity")

        self.assertEqual(event_type, "Contact:Phone-Write")
        self.assertEqual(role, "Participant")

    def test_maven_arg_mapping_returns_canonical_event_and_role(self) -> None:
        event_type = self.mapper.map_event_type("maven_arg", "Life:Die")
        role = self.mapper.map_role("maven_arg", "Life:Die", "Victim")

        self.assertEqual(event_type, "Life:Die")
        self.assertEqual(role, "Victim")

    def test_maven_arg_raw_labels_and_roles_map_when_safe(self) -> None:
        event_type = self.mapper.map_event_type("maven_arg", "Motion")
        role = self.mapper.map_role("maven_arg", "Attack", "Patient")

        self.assertEqual(event_type, "Movement:Transport")
        self.assertEqual(role, "Target")

    def test_swig_mapping_maps_common_verbs_into_canonical_ontology(self) -> None:
        event_type = self.mapper.map_event_type("swig", "attacking")
        role = self.mapper.map_role("swig", "attacking", "weapon")

        self.assertEqual(event_type, "Conflict:Attack")
        self.assertEqual(role, "Instrument")

    def test_swig_high_frequency_verbs_and_roles_map_when_safe(self) -> None:
        event_type = self.mapper.map_event_type("swig", "phoning")
        role = self.mapper.map_role("swig", "shooting", "firearm")

        self.assertEqual(event_type, "Contact:Phone-Write")
        self.assertEqual(role, "Instrument")

    def test_role_mapping_returns_none_when_role_cannot_be_mapped(self) -> None:
        role = self.mapper.map_role("swig", "attacking", "banana")

        self.assertIsNone(role)

    def test_uncertain_labels_remain_unmapped(self) -> None:
        self.assertIsNone(self.mapper.map_event_type("maven_arg", "Statement"))
        self.assertIsNone(self.mapper.map_event_type("swig", "destroying"))

    def test_unknown_dataset_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, 'unknown ontology mapping dataset "unknown_dataset"'):
            self.mapper.map_event_type("unknown_dataset", "Conflict:Attack")


if __name__ == "__main__":
    unittest.main()
