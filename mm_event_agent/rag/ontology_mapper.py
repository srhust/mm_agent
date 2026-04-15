"""Dataset-to-canonical ontology mapper.

`mm_event_agent.ontology` remains the single canonical ontology sink. Dataset
labels from ACE2005, MAVEN-ARG, and SWiG are translated through external JSON
mapping files so the mappings can be reviewed and expanded by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mm_event_agent.ontology import (
    get_allowed_roles,
    is_supported_event_type,
)

MAPPING_DIR = Path(__file__).resolve().parents[1] / "ontology_mappings"


class OntologyMapper:
    """Map dataset-specific event/role labels into the canonical ontology."""

    def __init__(self, mapping_dir: str | Path | None = None) -> None:
        self._mapping_dir = Path(mapping_dir) if mapping_dir is not None else MAPPING_DIR
        self._cache: dict[str, dict[str, Any]] = {}

    def map_event_type(self, dataset: str, event_type: str) -> str | None:
        """Return a canonical event type or None when no mapping exists."""
        normalized_dataset = self._normalize_key(dataset)
        normalized_event_type = self._normalize_key(event_type)
        if not normalized_event_type:
            return None

        mapping = self._load_mapping(normalized_dataset)
        mapped = mapping.get("event_type_map", {}).get(normalized_event_type)
        if mapped is None:
            return None
        if not is_supported_event_type(mapped):
            raise ValueError(
                f'mapping file "{normalized_dataset}" resolves to unsupported event_type "{mapped}"'
            )
        return mapped

    def map_role(self, dataset: str, event_type: str, role: str) -> str | None:
        """Return a canonical role or None when the mapping is missing/invalid."""
        normalized_role = self._normalize_key(role)
        if not normalized_role:
            return None

        canonical_event_type = self.map_event_type(dataset, event_type)
        if canonical_event_type is None:
            return None

        mapping = self._load_mapping(self._normalize_key(dataset))
        canonical_role = (
            mapping.get("role_map", {})
            .get(canonical_event_type, {})
            .get(normalized_role)
        )
        if canonical_role is None:
            return None
        if canonical_role not in get_allowed_roles(canonical_event_type):
            return None
        return canonical_role

    def _load_mapping(self, dataset: str) -> dict[str, Any]:
        cached = self._cache.get(dataset)
        if cached is not None:
            return cached

        path = self._mapping_dir / f"{dataset}.json"
        if not path.exists():
            raise ValueError(f'unknown ontology mapping dataset "{dataset}"')

        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f'mapping file "{path}" must contain a JSON object')

        event_type_map = raw.get("event_type_map", {})
        role_map = raw.get("role_map", {})
        if not isinstance(event_type_map, dict) or not isinstance(role_map, dict):
            raise ValueError(f'mapping file "{path}" must define object fields "event_type_map" and "role_map"')

        normalized = {
            "dataset": dataset,
            "canonical_ontology_module": str(raw.get("canonical_ontology_module") or "mm_event_agent.ontology"),
            "event_type_map": {
                self._normalize_key(key): str(value).strip()
                for key, value in event_type_map.items()
                if self._normalize_key(key) and str(value).strip()
            },
            "role_map": {
                str(event_type).strip(): {
                    self._normalize_key(role_key): str(role_value).strip()
                    for role_key, role_value in roles.items()
                    if self._normalize_key(role_key) and str(role_value).strip()
                }
                for event_type, roles in role_map.items()
                if str(event_type).strip() and isinstance(roles, dict)
            },
        }
        self._cache[dataset] = normalized
        return normalized

    @staticmethod
    def _normalize_key(value: Any) -> str:
        return str(value or "").strip().lower()
