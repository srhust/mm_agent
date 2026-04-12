"""Event ontology helpers for the supported M2E2-compatible closed set."""

from __future__ import annotations

from typing import TypedDict


class EventRoleSchema(TypedDict):
    definition: str
    trigger_hint: str
    text_roles: list[str]
    image_roles: list[str]
    role_definitions: dict[str, str]
    extraction_notes: list[str]


EVENT_ONTOLOGY: dict[str, EventRoleSchema] = {
    "Movement:Transport": {
        "definition": "A person or agent moves or transports an artifact or person from one place to another, often using a vehicle.",
        "trigger_hint": "transported, moved, carried, drove, flew, shipped, evacuated, taken to, arrived, departed",
        "text_roles": ["Agent", "Artifact", "Vehicle", "Destination", "Origin"],
        "image_roles": ["Agent", "Artifact", "Vehicle", "Destination", "Origin"],
        "role_definitions": {
            "Agent": "The mover, transporter, or controlling party responsible for the transport.",
            "Artifact": "The person or object being transported.",
            "Vehicle": "The vehicle or conveyance used for transport.",
            "Destination": "The place the artifact or person is transported to.",
            "Origin": "The place the artifact or person is transported from.",
        },
        "extraction_notes": [
            "Separate the mover from the thing being moved.",
            "Use Destination and Origin only when movement direction is grounded.",
            "In images, vehicles, carried objects, and travel context can support transport semantics.",
        ],
    },
    "Conflict:Attack": {
        "definition": "A violent or harmful attack is carried out against a target, person, group, or location.",
        "trigger_hint": "attack, attacked, bombed, exploded, fired on, shot, shelled, struck, assaulted",
        "text_roles": ["Attacker", "Target", "Instrument", "Place"],
        "image_roles": ["Attacker", "Target", "Instrument", "Place"],
        "role_definitions": {
            "Attacker": "The person, group, or force carrying out the attack.",
            "Target": "The person, group, object, or place that is attacked or damaged.",
            "Instrument": "The weapon, explosive, or means used in the attack.",
            "Place": "The location where the attack happens.",
        },
        "extraction_notes": [
            "Do not confuse the attack instrument with the attacker.",
            "Target can be people, structures, vehicles, or locations under attack.",
            "In images, damage, weapons, smoke, fire, or people under assault can support attack semantics, but keep grounding unresolved unless a bbox is already available.",
        ],
    },
    "Conflict:Demonstrate": {
        "definition": "A public protest, rally, or demonstration occurs, often involving demonstrators and police presence.",
        "trigger_hint": "protested, demonstrated, rallied, marched, gathered, clashed, unrest, demonstration",
        "text_roles": ["Entity", "Police", "Instrument", "Place"],
        "image_roles": ["Entity", "Police", "Instrument", "Place"],
        "role_definitions": {
            "Entity": "The demonstrators, protesters, crowd, or participating group.",
            "Police": "Police or security forces present in the demonstration context.",
            "Instrument": "Objects used in the demonstration context such as signs, banners, shields, or crowd-control means.",
            "Place": "The location where the demonstration occurs.",
        },
        "extraction_notes": [
            "Entity usually refers to protesters or organized groups rather than any bystander.",
            "Police should only be used when law enforcement or security presence is actually indicated.",
            "In images, crowds, protest signs, riot gear, barricades, and street gathering context can support this event type.",
        ],
    },
    "Justice:Arrest-Jail": {
        "definition": "A person is arrested, detained, taken into custody, or jailed by an authority.",
        "trigger_hint": "arrested, detained, taken into custody, jailed, imprisoned, booked, handcuffed",
        "text_roles": ["Agent", "Person", "Instrument", "Place"],
        "image_roles": ["Agent", "Person", "Instrument", "Place"],
        "role_definitions": {
            "Agent": "The arresting or detaining authority, such as police or security personnel.",
            "Person": "The person being arrested, detained, or jailed.",
            "Instrument": "The means used in the arrest context, such as handcuffs, restraints, or official vehicle.",
            "Place": "The location where the arrest or jailing occurs.",
        },
        "extraction_notes": [
            "Agent is the authority; Person is the detainee.",
            "Instrument is optional and should not be invented from generic law-enforcement context.",
            "In images, officers restraining a person, handcuffs, or custody scenes can support this event type.",
        ],
    },
    "Contact:Phone-Write": {
        "definition": "A communication occurs through phone, writing, messaging, or other mediated contact rather than an in-person meeting.",
        "trigger_hint": "called, phoned, wrote, messaged, emailed, texted, posted, communicated",
        "text_roles": ["Entity", "Instrument", "Place"],
        "image_roles": ["Entity", "Instrument", "Place"],
        "role_definitions": {
            "Entity": "The communicating participant or participants.",
            "Instrument": "The communication medium or device, such as a phone, letter, message, or computer.",
            "Place": "The location associated with the communication when grounded.",
        },
        "extraction_notes": [
            "Use this for mediated communication, not face-to-face interaction.",
            "Instrument can be a communication device or medium when supported.",
            "In images, phones, screens, written documents, or messaging context can support this event type.",
        ],
    },
    "Contact:Meet": {
        "definition": "People or parties meet in person or gather together for direct interaction.",
        "trigger_hint": "met, meeting, met with, gathered, talks, summit, conference, visited",
        "text_roles": ["Participant", "Place"],
        "image_roles": ["Participant", "Place"],
        "role_definitions": {
            "Participant": "A person or group participating in the meeting.",
            "Place": "The location where the meeting occurs.",
        },
        "extraction_notes": [
            "Reserve this event type for in-person or co-present interaction rather than mediated contact.",
            "Participant can cover multiple people or groups.",
            "In images, people seated together, handshakes, conference tables, or meeting rooms can support this event type.",
        ],
    },
    "Life:Die": {
        "definition": "A person dies, is killed, or is found dead, whether from violence, disaster, or another cause.",
        "trigger_hint": "died, killed, dead, slain, perished, fatalities, was shot dead, lost his life",
        "text_roles": ["Victim", "Agent", "Instrument", "Place"],
        "image_roles": ["Victim", "Agent", "Instrument", "Place"],
        "role_definitions": {
            "Victim": "The person who dies.",
            "Agent": "The person, group, or force causing the death when specified.",
            "Instrument": "The weapon, means, or cause of death when specified.",
            "Place": "The location where the death occurs.",
        },
        "extraction_notes": [
            "Victim is central; Agent and Instrument are optional and should only be used when grounded.",
            "Do not force an Agent when the death is accidental or unspecified.",
            "In images, bodies, funerary context, or visible lethal aftermath may support the event, but unresolved semantic candidates are still allowed.",
        ],
    },
    "Transaction:Transfer-Money": {
        "definition": "Money is given, paid, transferred, donated, or otherwise moved from one party to another.",
        "trigger_hint": "paid, donated, transferred, funded, gave money, compensation, payment, remittance",
        "text_roles": ["Giver", "Recipient", "Money"],
        "image_roles": ["Giver", "Recipient", "Money"],
        "role_definitions": {
            "Giver": "The person, group, or organization providing the money.",
            "Recipient": "The person, group, or organization receiving the money.",
            "Money": "The amount, funds, or financial asset being transferred.",
        },
        "extraction_notes": [
            "Use this for money transfer, donation, payment, or funding events.",
            "Do not infer Money unless an amount or monetary reference is actually supported.",
            "In images, cash, payment handover, financial documents, or donation context can provide semantic support.",
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


def get_event_schema(event_type: str) -> EventRoleSchema | None:
    schema = EVENT_ONTOLOGY.get(event_type)
    if schema is None:
        return None
    return {
        "definition": schema["definition"],
        "trigger_hint": schema["trigger_hint"],
        "text_roles": list(schema["text_roles"]),
        "image_roles": list(schema["image_roles"]),
        "role_definitions": dict(schema["role_definitions"]),
        "extraction_notes": list(schema["extraction_notes"]),
    }


def format_event_schema_for_prompt(event_type: str) -> str:
    schema = get_event_schema(event_type)
    if schema is None:
        return "(unknown event type)"
    role_lines = [
        f"- {role}: {schema['role_definitions'][role]}"
        for role in schema["role_definitions"]
    ]
    note_lines = [f"- {note}" for note in schema["extraction_notes"]]
    return (
        f"event_type: {event_type}\n"
        f"definition: {schema['definition']}\n"
        f"trigger_hint: {schema['trigger_hint']}\n"
        f"allowed text roles: {json_list(schema['text_roles'])}\n"
        f"allowed image roles: {json_list(schema['image_roles'])}\n"
        "role_definitions:\n"
        f"{chr(10).join(role_lines)}\n"
        "extraction_notes:\n"
        f"{chr(10).join(note_lines)}"
    )


def format_full_ontology_for_prompt() -> str:
    return "\n\n".join(
        format_event_schema_for_prompt(event_type)
        for event_type in get_supported_event_types()
    )


def json_list(values: list[str]) -> str:
    return "[" + ", ".join(values) + "]"
