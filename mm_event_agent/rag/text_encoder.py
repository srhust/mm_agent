"""Reusable dense text encoder for persistent text-side RAG."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence

import numpy as np

try:  # pragma: no cover - exercised via mocks in unit tests
    import torch
    import torch.nn.functional as F
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
except ImportError:  # pragma: no cover - dependency availability varies by environment
    torch = None
    F = None
    Cache = None
    ModelOutput = object
    Qwen3VLConfig = None
    Qwen3VLModel = None
    Qwen3VLPreTrainedModel = object
    Qwen3VLProcessor = None
    Unpack = object
    TransformersKwargs = object

try:  # pragma: no cover - exercised via mocks in unit tests
    from qwen_vl_utils.vision_process import process_vision_info
except ImportError:  # pragma: no cover - dependency availability varies by environment
    process_vision_info = None


def _has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _resolve_torch_dtype(dtype: str):
    if torch is None:
        raise ImportError("torch is required to use Qwen3VLTextEncoder")
    normalized = str(dtype or "bfloat16").strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported torch dtype: {dtype}")
    return mapping[normalized]


class Qwen3VLForEmbeddingOutput(ModelOutput):  # pragma: no cover - thin container
    last_hidden_state: "torch.FloatTensor | None" = None
    attention_mask: "torch.Tensor | None" = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):  # pragma: no cover - exercised via backend mocks
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: "Qwen3VLConfig"

    def __init__(self, config: "Qwen3VLConfig"):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def forward(
        self,
        input_ids: "torch.LongTensor" = None,
        attention_mask: "torch.Tensor | None" = None,
        position_ids: "torch.LongTensor | None" = None,
        past_key_values: "Cache | None" = None,
        inputs_embeds: "torch.FloatTensor | None" = None,
        pixel_values: "torch.Tensor | None" = None,
        pixel_values_videos: "torch.FloatTensor | None" = None,
        image_grid_thw: "torch.LongTensor | None" = None,
        video_grid_thw: "torch.LongTensor | None" = None,
        cache_position: "torch.LongTensor | None" = None,
        logits_to_keep=0,
        **kwargs: "Unpack[TransformersKwargs]",
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


class _Qwen3VLTextModel:
    """Reference-aligned text-only Qwen3-VL embedding backend."""

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
                "Qwen3VLTextEncoder requires torch, transformers with qwen3_vl support, and qwen-vl-utils"
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

    def encode_texts(self, texts: Sequence[str], *, instruction: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        conversations = [self._format_text_conversation(str(text), instruction=instruction) for text in texts]
        model_inputs = self._preprocess_inputs(conversations)
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
        with torch.no_grad():
            outputs = self.model(**model_inputs)
            embeddings = self._pool_last_valid_token(
                outputs.last_hidden_state,
                model_inputs["attention_mask"],
            )
        return embeddings.detach().to("cpu").float().numpy().astype(np.float32, copy=False)

    def _format_text_conversation(self, text: str, *, instruction: str) -> list[dict[str, object]]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": str(instruction or "").strip() or "Represent the user's input."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": str(text)}],
            },
        ]

    def _preprocess_inputs(self, conversations: list[list[dict[str, object]]]) -> dict[str, "torch.Tensor"]:
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


class Qwen3VLTextEncoder:
    """Text-only encoder compatible with the local Qwen3-VL embedding contract."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_impl: str = "sdpa",
        instruction: str = "Retrieve text relevant to the user's query.",
        out_dim: int = 0,
        normalize: bool = True,
    ) -> None:
        self.model_path = str(model_path or "").strip()
        self.device = str(device or "cuda:0")
        self.dtype = str(dtype or "bfloat16")
        self.attn_impl = str(attn_impl or "sdpa")
        self.instruction = str(instruction or "Retrieve text relevant to the user's query.")
        self.out_dim = max(0, int(out_dim))
        self.normalize = bool(normalize)
        self._backend = _Qwen3VLTextModel(
            model_path=self.model_path,
            device=self.device,
            dtype=self.dtype,
            attn_impl=self.attn_impl,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        array = np.asarray(
            self._backend.encode_texts([str(text) for text in texts], instruction=self.instruction),
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
        return array.astype(np.float32, copy=False)
