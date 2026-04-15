"""Reusable dense image encoder for persistent SWiG image-side RAG."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from mm_event_agent.rag.text_encoder import (
    Qwen3VLForEmbedding,
    Qwen3VLProcessor,
    _has_flash_attn,
    _resolve_torch_dtype,
    process_vision_info,
    torch,
)


def verify_image_path(path: str) -> None:
    normalized = str(path or "").strip()
    if not normalized:
        raise ValueError("image path must be a non-empty string")
    if not os.path.exists(normalized):
        raise FileNotFoundError(f"image path does not exist: {normalized}")
    try:
        with Image.open(normalized) as image:
            image.verify()
    except Exception as exc:  # pragma: no cover - depends on image codec/runtime files
        raise ValueError(f"unreadable image: {normalized}") from exc


class _Qwen3VLImageModel:
    """Reference-aligned image-only Qwen3-VL embedding backend."""

    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        dtype: str,
        attn_impl: str,
    ) -> None:
        if torch is None or Qwen3VLProcessor is None or process_vision_info is None:
            raise ImportError(
                "Qwen3VLImageEncoder requires torch, transformers with qwen3_vl support, and qwen-vl-utils"
            )

        self.model_path = str(model_path or "").strip()
        if not self.model_path:
            raise ValueError("model_path must be a non-empty local filesystem path")
        model_root = Path(self.model_path)
        if not model_root.exists():
            raise FileNotFoundError(f"Qwen3-VL model path does not exist: {model_root}")

        resolved_attn = str(attn_impl or "sdpa").strip() or "sdpa"
        if resolved_attn == "flash_attention_2" and not _has_flash_attn():
            resolved_attn = "sdpa"

        self.device = torch.device(str(device or "cuda:0"))
        self.model = Qwen3VLForEmbedding.from_pretrained(
            self.model_path,
            torch_dtype=_resolve_torch_dtype(dtype),
            attn_implementation=resolved_attn,
            local_files_only=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.processor = Qwen3VLProcessor.from_pretrained(
            self.model_path,
            padding_side="right",
            local_files_only=True,
        )

    def encode_image_paths(
        self,
        image_paths: Sequence[str],
        *,
        instruction: str,
        batch_size: int = 8,
    ) -> np.ndarray:
        if not image_paths:
            return np.zeros((0, 0), dtype=np.float32)

        normalized_paths = [str(path) for path in image_paths]
        effective_batch_size = max(1, int(batch_size))
        batches: list[np.ndarray] = []
        for start in range(0, len(normalized_paths), effective_batch_size):
            batches.append(
                self._encode_image_batch(
                    normalized_paths[start : start + effective_batch_size],
                    instruction=instruction,
                )
            )
        return np.concatenate(batches, axis=0) if batches else np.zeros((0, 0), dtype=np.float32)

    def _encode_image_batch(
        self,
        image_paths: Sequence[str],
        *,
        instruction: str,
    ) -> np.ndarray:
        conversations = [self._format_image_conversation(path, instruction=instruction) for path in image_paths]
        model_inputs = self._preprocess_inputs(conversations)
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
        with torch.no_grad():
            outputs = self.model(**model_inputs)
            embeddings = self._pool_last_valid_token(outputs.last_hidden_state, model_inputs["attention_mask"])
        return embeddings.detach().to("cpu").float().numpy().astype(np.float32, copy=False)

    def _format_image_conversation(self, image_path: str, *, instruction: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": str(instruction or "").strip() or "Represent the user's input."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "file://" + str(image_path),
                    }
                ],
            },
        ]

    def _preprocess_inputs(self, conversations: list[list[dict[str, Any]]]) -> dict[str, "torch.Tensor"]:
        texts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]
        try:
            vision_outputs = process_vision_info(conversations, image_patch_size=16)
        except TypeError:
            vision_outputs = process_vision_info(conversations)
        if not isinstance(vision_outputs, tuple) or len(vision_outputs) not in {2, 3}:
            raise RuntimeError("unexpected process_vision_info output")
        images = vision_outputs[0]
        videos = vision_outputs[1]
        extra_kwargs = vision_outputs[2] if len(vision_outputs) == 3 and isinstance(vision_outputs[2], dict) else {}
        return self.processor(
            text=texts,
            images=images,
            videos=videos,
            truncation=True,
            padding=True,
            return_tensors="pt",
            **extra_kwargs,
        )

    @staticmethod
    def _pool_last_valid_token(hidden_state: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        flipped = attention_mask.flip(dims=[1])
        last_one = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_one - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]


class Qwen3VLImageEncoder:
    """Image-path encoder compatible with the local Qwen3-VL embedding contract."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_impl: str = "sdpa",
        instruction: str = "Retrieve images relevant to the user's query.",
        out_dim: int = 0,
        normalize: bool = True,
    ) -> None:
        self.model_path = str(model_path or "").strip()
        self.device = str(device or "cuda:0")
        self.dtype = str(dtype or "bfloat16")
        self.attn_impl = str(attn_impl or "sdpa")
        self.instruction = str(instruction or "Retrieve images relevant to the user's query.")
        self.out_dim = max(0, int(out_dim))
        self.normalize = bool(normalize)
        self._backend = _Qwen3VLImageModel(
            model_path=self.model_path,
            device=self.device,
            dtype=self.dtype,
            attn_impl=self.attn_impl,
        )

    def encode_image_paths(self, image_paths: Sequence[str], batch_size: int = 8) -> np.ndarray:
        normalized_paths = [str(path) for path in image_paths]
        if not normalized_paths:
            return np.zeros((0, 0), dtype=np.float32)
        for path in normalized_paths:
            verify_image_path(path)
        array = np.asarray(
            self._backend.encode_image_paths(
                normalized_paths,
                instruction=self.instruction,
                batch_size=max(1, int(batch_size)),
            ),
            dtype=np.float32,
        )
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(f"expected 2D embeddings, got shape {array.shape}")
        if self.out_dim > 0 and array.shape[1] > self.out_dim:
            array = array[:, : self.out_dim]
        if self.normalize and array.size:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            array = array / norms
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return array.astype(np.float32, copy=False)
