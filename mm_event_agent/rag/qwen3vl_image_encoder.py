"""Reusable Qwen3-VL image encoder for local RAG indexing.

This module preserves the embedding contract used by the reference script:

- model wrapper: `Qwen3VLModel` exposed through a thin embedding wrapper
- preprocessing: `Qwen3VLProcessor` + `process_vision_info`
- pooling: last valid token from `last_hidden_state`
- normalization: optional L2 normalization after pooling
- output slicing: optional prefix slice for MRL-style smaller dimensions

The class is intended to be imported from Python code and reused by offline
index builders that later persist vectors into FAISS.
"""

from __future__ import annotations

import importlib.util
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils.vision_process import process_vision_info
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1.0
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS
DEFAULT_INSTRUCTION = "Retrieve images relevant to the user's query."


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Thin wrapper around Qwen3VLModel that returns token states for pooling."""

    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Qwen3VLForEmbeddingOutput:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


def _is_image_path(path: str) -> bool:
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"}
    clean_path = urlparse(path).path if path.startswith(("http://", "https://")) else path
    _, ext = os.path.splitext(clean_path.lower())
    return ext in image_extensions


def _is_video_input(video: Any) -> bool:
    if isinstance(video, str):
        return True
    if isinstance(video, list) and video:
        first = video[0]
        if isinstance(first, Image.Image):
            return True
        if isinstance(first, str):
            return _is_image_path(first)
    return False


def _has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def verify_image(path: str | os.PathLike[str]) -> Optional[str]:
    """Return an error string for an unreadable image, else None."""
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except Exception as exc:  # pragma: no cover - passthrough for CLI diagnostics
        return f"{type(exc).__name__}:{exc}"


class Qwen3VLImageEncoder:
    """Reusable path-based image encoder preserving the reference embed contract."""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        *,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = DEFAULT_INSTRUCTION,
        attn_implementation: str = "sdpa",
        **from_pretrained_kwargs: Any,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction

        if attn_implementation == "flash_attention_2" and not _has_flash_attn():
            attn_implementation = "sdpa"

        pretrained_kwargs = dict(from_pretrained_kwargs)
        pretrained_kwargs["attn_implementation"] = attn_implementation
        pretrained_kwargs.setdefault("trust_remote_code", True)

        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path,
            **pretrained_kwargs,
        ).to(self.device)
        self.model.eval()

        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path,
            padding_side="right",
        )

    @staticmethod
    def _ensure_instruction_punct(instruction: str) -> str:
        value = instruction.strip()
        if value and not unicodedata.category(value[-1]).startswith("P"):
            value = value + "."
        return value

    def format_model_input(
        self,
        *,
        text: Optional[Union[Sequence[str], str]] = None,
        image: Optional[Union[Sequence[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[Union[Sequence[Any], Any]] = None,
        instruction: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        prompt = self._ensure_instruction_punct(instruction) if instruction else self.default_instruction

        texts = [text] if isinstance(text, str) else list(text or [])
        images = [image] if isinstance(image, (str, Image.Image)) else list(image or [])
        if video is None:
            videos: list[Any] = []
        elif _is_video_input(video):
            videos = [video]
        else:
            videos = list(video)

        content: list[dict[str, Any]] = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": content},
        ]

        if not texts and not images and not videos:
            content.append({"type": "text", "text": "NULL"})
            return conversation

        for value in images:
            if isinstance(value, Image.Image):
                image_content: str | Image.Image = value
            elif isinstance(value, str):
                image_content = value if value.startswith(("http://", "https://")) else ("file://" + value)
            else:  # pragma: no cover - guarded by caller types
                raise TypeError(f"Unrecognized image type: {type(value)}")
            content.append(
                {
                    "type": "image",
                    "image": image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )

        for value in texts:
            content.append({"type": "text", "text": value})

        return conversation

    def _preprocess_inputs(self, conversations: list[list[dict[str, Any]]]) -> dict[str, torch.Tensor]:
        texts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]

        try:
            processed = process_vision_info(conversations, image_patch_size=16)
        except TypeError:
            processed = process_vision_info(conversations)

        if not isinstance(processed, tuple):
            raise RuntimeError(f"process_vision_info returned unexpected type: {type(processed)}")

        images = None
        videos = None
        video_kwargs: dict[str, Any] = {}
        if len(processed) == 2:
            images, videos = processed
        elif len(processed) == 3:
            images, videos, video_kwargs = processed
        else:
            raise RuntimeError(f"process_vision_info returned unexpected tuple length: {len(processed)}")

        return self.processor(
            text=texts,
            images=images,
            videos=videos,
            video_metadata=None,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=True,
            return_tensors="pt",
            **video_kwargs,
        )

    @staticmethod
    def _pool_last_valid_token(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        flipped = attention_mask.flip(dims=[1])
        last_one = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_one - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    @torch.no_grad()
    def encode_inputs(
        self,
        inputs: Sequence[dict[str, Any]],
        *,
        normalize: bool = True,
        out_dim: Optional[int] = None,
    ) -> np.ndarray:
        conversations = [
            self.format_model_input(
                text=item.get("text"),
                image=item.get("image"),
                instruction=item.get("instruction"),
            )
            for item in inputs
        ]
        model_inputs = self._preprocess_inputs(conversations)
        model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}

        outputs = self.model(**model_inputs)
        embeddings = self._pool_last_valid_token(outputs.last_hidden_state, model_inputs["attention_mask"])
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        array = embeddings.detach().to("cpu").float().numpy()
        if out_dim and 0 < out_dim < array.shape[1]:
            array = array[:, :out_dim]
        return array

    def encode_image_paths(
        self,
        image_paths: Sequence[str | os.PathLike[str]],
        *,
        instruction: Optional[str] = None,
        normalize: bool = True,
        out_dim: Optional[int] = None,
    ) -> np.ndarray:
        inputs = [
            {
                "image": str(Path(path)),
                "instruction": instruction or self.default_instruction,
            }
            for path in image_paths
        ]
        return self.encode_inputs(inputs, normalize=normalize, out_dim=out_dim)

    def encode_image_path(
        self,
        image_path: str | os.PathLike[str],
        *,
        instruction: Optional[str] = None,
        normalize: bool = True,
        out_dim: Optional[int] = None,
    ) -> np.ndarray:
        return self.encode_image_paths(
            [image_path],
            instruction=instruction,
            normalize=normalize,
            out_dim=out_dim,
        )
