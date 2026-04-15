"""Dataset normalizers for future persistent layered-RAG corpora."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mm_event_agent.rag.ontology_mapper import OntologyMapper


def clean_text(value: Any) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()
    return " ".join(text.split())


def build_retrieval_text(*parts: Any) -> str:
    flattened: list[str] = []
    for part in parts:
        if isinstance(part, list):
            flattened.extend(clean_text(item) for item in part if clean_text(item))
        else:
            value = clean_text(part)
            if value:
                flattened.append(value)
    return " ".join(flattened)


def make_skip_reason(reason: str, detail: str = "") -> str:
    normalized_reason = clean_text(reason).lower().replace(" ", "_")
    normalized_detail = clean_text(detail).lower().replace(" ", "_")
    return normalized_reason if not normalized_detail else f"{normalized_reason}:{normalized_detail}"


def increment_skip_reason(counter: Counter[str], reason: str) -> str:
    counter[reason] += 1
    return reason


@dataclass
class SwigNormalizedRecord:
    text_example: dict
    image_manifest: dict


@dataclass
class _BaseNormalizer:
    mapper: OntologyMapper
    dataset_name: str
    skipped_by_reason: Counter[str] = field(default_factory=Counter)
    last_skip_reason: str | None = None

    def _skip(self, reason: str, detail: str = "") -> None:
        self.last_skip_reason = increment_skip_reason(self.skipped_by_reason, make_skip_reason(reason, detail))

    def _keep(self) -> None:
        self.last_skip_reason = None

    @staticmethod
    def _get_first(raw: dict[str, Any], keys: list[str], default: Any = "") -> Any:
        for key in keys:
            if key in raw and raw.get(key) is not None:
                return raw.get(key)
        return default

    @staticmethod
    def _extract_event_type(raw: dict[str, Any]) -> str:
        direct = clean_text(
            _BaseNormalizer._get_first(raw, ["event_type", "type", "label", "event_label", "verb"], "")
        )
        subtype = clean_text(_BaseNormalizer._get_first(raw, ["subtype"], ""))
        if direct and subtype and ":" not in direct:
            return f"{direct}:{subtype}"
        return direct

    @staticmethod
    def _extract_text(raw: dict[str, Any]) -> str:
        text = _BaseNormalizer._get_first(raw, ["raw_text", "text", "sentence", "source_text"], "")
        if isinstance(text, list):
            return clean_text(" ".join(str(item) for item in text))
        return clean_text(text)

    @staticmethod
    def _extract_trigger(raw: dict[str, Any], raw_text: str) -> dict[str, Any] | None:
        trigger = raw.get("trigger")
        if isinstance(trigger, dict):
            trigger_text = clean_text(trigger.get("text"))
            span = trigger.get("span")
            if not isinstance(span, dict):
                start = trigger.get("start")
                end = trigger.get("end")
                span = {"start": start, "end": end} if isinstance(start, int) and isinstance(end, int) else None
            if trigger_text:
                return {"text": trigger_text, "span": span if isinstance(span, dict) else None}
        trigger_text = clean_text(_BaseNormalizer._get_first(raw, ["trigger_text", "trigger", "mention"], ""))
        start = raw.get("trigger_start")
        end = raw.get("trigger_end")
        span = {"start": start, "end": end} if isinstance(start, int) and isinstance(end, int) else None
        if trigger_text:
            return {"text": trigger_text, "span": span}
        if raw_text:
            return None
        return None

    def _normalize_text_arguments(
        self,
        raw_arguments: Any,
        raw_event_type: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_arguments, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in raw_arguments:
            if not isinstance(item, dict):
                continue
            mapped_role = self.mapper.map_role(
                self.dataset_name,
                raw_event_type,
                self._get_first(item, ["role", "argument_role", "semantic_role"], ""),
            )
            if mapped_role is None:
                continue
            text = clean_text(self._get_first(item, ["text", "mention", "label", "argument"], ""))
            if not text:
                continue
            span = item.get("span")
            if not isinstance(span, dict):
                start = item.get("start")
                end = item.get("end")
                span = {"start": start, "end": end} if isinstance(start, int) and isinstance(end, int) else None
            normalized.append({"role": mapped_role, "text": text, "span": span if isinstance(span, dict) else None})
        return normalized

    def _normalize_image_arguments(
        self,
        raw_arguments: Any,
        raw_event_type: str,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        if isinstance(raw_arguments, dict):
            iterable = [
                {"role": role, "label": value}
                for role, value in raw_arguments.items()
            ]
        elif isinstance(raw_arguments, list):
            iterable = raw_arguments
        else:
            iterable = []

        for item in iterable:
            if not isinstance(item, dict):
                continue
            mapped_role = self.mapper.map_role(
                self.dataset_name,
                raw_event_type,
                self._get_first(item, ["role", "semantic_role"], ""),
            )
            if mapped_role is None:
                continue
            label = clean_text(self._get_first(item, ["label", "text", "noun", "value"], ""))
            if not label:
                continue
            normalized.append({"role": mapped_role, "label": label})
        return normalized


class Ace2005Normalizer(_BaseNormalizer):
    def __init__(self, mapper: OntologyMapper) -> None:
        super().__init__(mapper=mapper, dataset_name="ace2005")

    def normalize(self, raw_sample: dict[str, Any]) -> dict[str, Any] | None:
        raw_event_type = self._extract_event_type(raw_sample)
        canonical_event_type = self.mapper.map_event_type(self.dataset_name, raw_event_type)
        if canonical_event_type is None:
            self._skip("unmapped_event_type", raw_event_type or "missing")
            return None

        raw_text = self._extract_text(raw_sample)
        if not raw_text:
            self._skip("missing_raw_text")
            return None

        record_id = clean_text(self._get_first(raw_sample, ["id", "event_id"], "")) or f"ace2005::{clean_text(self._get_first(raw_sample, ['doc_id', 'docid'], 'unknown'))}"
        doc_id = clean_text(self._get_first(raw_sample, ["doc_id", "docid", "document_id"], ""))
        trigger = self._extract_trigger(raw_sample, raw_text)
        if trigger is None or not clean_text(trigger.get("text")):
            self._skip("missing_trigger")
            return None

        text_arguments = self._normalize_text_arguments(
            self._get_first(raw_sample, ["text_arguments", "arguments", "args"], []),
            raw_event_type,
        )
        pattern_summary = build_retrieval_text(
            canonical_event_type,
            trigger.get("text"),
            [f"{arg['role']} {arg['text']}" for arg in text_arguments],
        )
        record = {
            "id": record_id,
            "source_dataset": "ACE2005",
            "doc_id": doc_id,
            "modality": "text",
            "event_type": canonical_event_type,
            "trigger": trigger,
            "raw_text": raw_text,
            "text_arguments": text_arguments,
            "pattern_summary": pattern_summary,
            "retrieval_text": build_retrieval_text(
                canonical_event_type,
                trigger.get("text"),
                raw_text,
                pattern_summary,
                [arg["role"] for arg in text_arguments],
                [arg["text"] for arg in text_arguments],
            ),
        }
        self._keep()
        return record


class MavenArgNormalizer(_BaseNormalizer):
    def __init__(self, mapper: OntologyMapper) -> None:
        super().__init__(mapper=mapper, dataset_name="maven_arg")

    def normalize(self, raw_sample: dict[str, Any]) -> dict[str, Any] | None:
        raw_event_type = self._extract_event_type(raw_sample)
        canonical_event_type = self.mapper.map_event_type(self.dataset_name, raw_event_type)
        if canonical_event_type is None:
            self._skip("unmapped_event_type", raw_event_type or "missing")
            return None

        raw_text = self._extract_text(raw_sample)
        if not raw_text:
            self._skip("missing_raw_text")
            return None

        record_id = clean_text(self._get_first(raw_sample, ["id", "event_id", "mention_id"], "")) or f"maven_arg::{clean_text(self._get_first(raw_sample, ['doc_id', 'sentence_id'], 'unknown'))}"
        doc_id = clean_text(self._get_first(raw_sample, ["doc_id", "document_id", "sentence_id"], ""))
        trigger = self._extract_trigger(raw_sample, raw_text)
        if trigger is None or not clean_text(trigger.get("text")):
            self._skip("missing_trigger")
            return None

        text_arguments = self._normalize_text_arguments(
            self._get_first(raw_sample, ["text_arguments", "arguments", "args"], []),
            raw_event_type,
        )
        pattern_summary = build_retrieval_text(
            canonical_event_type,
            trigger.get("text"),
            [f"{arg['role']} {arg['text']}" for arg in text_arguments],
        )
        record = {
            "id": record_id,
            "source_dataset": "MAVEN-ARG",
            "doc_id": doc_id,
            "modality": "text",
            "event_type": canonical_event_type,
            "trigger": trigger,
            "raw_text": raw_text,
            "text_arguments": text_arguments,
            "pattern_summary": pattern_summary,
            "retrieval_text": build_retrieval_text(
                canonical_event_type,
                trigger.get("text"),
                raw_text,
                pattern_summary,
                [arg["role"] for arg in text_arguments],
                [arg["text"] for arg in text_arguments],
            ),
        }
        self._keep()
        return record


class SwigNormalizer(_BaseNormalizer):
    def __init__(self, mapper: OntologyMapper) -> None:
        super().__init__(mapper=mapper, dataset_name="swig")

    def normalize(self, raw_sample: dict[str, Any], *, images_root: str | Path | None = None) -> SwigNormalizedRecord | None:
        raw_event_type = clean_text(self._get_first(raw_sample, ["swig_verb", "verb", "event_type"], ""))
        canonical_event_type = self.mapper.map_event_type(self.dataset_name, raw_event_type)
        if canonical_event_type is None:
            self._skip("unmapped_event_type", raw_event_type or "missing")
            return None

        image_id = clean_text(self._get_first(raw_sample, ["image_id", "id", "img_id"], ""))
        if not image_id:
            self._skip("missing_image_id")
            return None

        image_desc = clean_text(self._get_first(raw_sample, ["image_desc", "caption", "description"], ""))
        visual_situation = clean_text(self._get_first(raw_sample, ["visual_situation", "situation", "frame_description"], "")) or raw_event_type
        swig_frame = clean_text(self._get_first(raw_sample, ["swig_frame", "frame"], ""))
        image_arguments = self._normalize_image_arguments(
            self._get_first(raw_sample, ["image_arguments", "arguments", "roles", "nouns"], []),
            raw_event_type,
        )

        relative_path = clean_text(self._get_first(raw_sample, ["path", "image_path", "file_name", "filename"], ""))
        if not relative_path:
            self._skip("missing_path")
            return None
        if images_root is not None and relative_path:
            resolved_path = str((Path(images_root) / relative_path).resolve())
        else:
            resolved_path = relative_path

        visual_pattern_summary = build_retrieval_text(
            canonical_event_type,
            raw_event_type,
            visual_situation,
            [f"{arg['role']} {arg['label']}" for arg in image_arguments],
        )
        text_example = {
            "id": f"swig-text::{image_id}",
            "source_dataset": "SWiG",
            "image_id": image_id,
            "modality": "image_semantics",
            "event_type": canonical_event_type,
            "swig_verb": raw_event_type,
            "swig_frame": swig_frame,
            "image_desc": image_desc,
            "visual_situation": visual_situation,
            "image_arguments": image_arguments,
            "visual_pattern_summary": visual_pattern_summary,
            "retrieval_text": build_retrieval_text(
                canonical_event_type,
                raw_event_type,
                swig_frame,
                image_desc,
                visual_situation,
                visual_pattern_summary,
                [arg["role"] for arg in image_arguments],
                [arg["label"] for arg in image_arguments],
            ),
        }
        image_manifest = {
            "id": f"swig-image::{image_id}",
            "source_dataset": "SWiG",
            "image_id": image_id,
            "path": resolved_path,
            "event_type": canonical_event_type,
            "swig_verb": raw_event_type,
            "swig_frame": swig_frame,
            "image_arguments": image_arguments,
            "retrieval_text": build_retrieval_text(
                canonical_event_type,
                raw_event_type,
                swig_frame,
                [arg["role"] for arg in image_arguments],
                [arg["label"] for arg in image_arguments],
            ),
        }
        self._keep()
        return SwigNormalizedRecord(text_example=text_example, image_manifest=image_manifest)
