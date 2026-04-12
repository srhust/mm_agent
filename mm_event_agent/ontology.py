"""Event ontology helpers for the current supported M2E2-compatible subset."""

from __future__ import annotations

from typing import TypedDict


class EventRoleSchema(TypedDict):
    text_roles: list[str]
    image_roles: list[str]


# Current project subset: only the explosion event type is wired through examples/RAG.
# It is ACE2005-compatible and treated as the current closed set.
EVENT_ONTOLOGY: dict[str, EventRoleSchema] = {
    "explosion": {
        "text_roles": [
            "location",
            "place",
            "device",
            "cause",
            "perpetrator",
            "victim",
            "casualties",
            "time",
        ],
        "image_roles": [
            "smoke",
            "fire",
            "damage",
            "victim",
            "vehicle",
            "building",
            "crowd",
            "location",
        ],
    }
}


def get_supported_event_types() -> list[str]:
    return list(EVENT_ONTOLOGY.keys())


def is_supported_event_type(event_type: str) -> bool:
    return event_type in EVENT_ONTOLOGY


def get_allowed_roles(event_type: str) -> list[str]:
    schema = EVENT_ONTOLOGY.get(event_type)
    if schema is None:
        return []
    return list(dict.fromkeys(schema["text_roles"] + schema["image_roles"]))


def get_allowed_text_roles(event_type: str) -> list[str]:
    schema = EVENT_ONTOLOGY.get(event_type)
    return list(schema["text_roles"]) if schema is not None else []


def get_allowed_image_roles(event_type: str) -> list[str]:
    schema = EVENT_ONTOLOGY.get(event_type)
    return list(schema["image_roles"]) if schema is not None else []
