#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embed_images_shard.py
- Read a shard JSONL produced by split_manifest.py
- Embed images with Qwen3-VL-Embedding using Transformers + local checkpoint
- Save embeddings as float16 .npy + an ids mapping file
- Log bad/failed images to TSV

Example:
python embed_images_shard.py \
  --shard /home/DATA2/hwxdataset/shards/shard_000.jsonl \
  --out_dir /home/DATA2/hwxdataset/emb_shards \
  --model_path /home/DATA4/hwx/Qwen3-VL-embedding \
  --device cuda:0 \
  --batch_size 16 \
  --max_items -1 \
  --instruction "Retrieve images relevant to the user's query." \
  --out_dim 4096 \
  --normalize
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

# ---- Qwen3-VL Embedding: Transformers + qwen-vl-utils (inlined) ----
# Requires:
#   transformers>=4.57.0
#   qwen-vl-utils>=0.0.14
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLPreTrainedModel,
    Qwen3VLModel,
    Qwen3VLConfig,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import Cache
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack

from qwen_vl_utils.vision_process import process_vision_info

import unicodedata
from urllib.parse import urlparse
from dataclasses import dataclass


# Defaults (follow official repo defaults)
MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1.0
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """A thin wrapper around Qwen3VLModel that returns last_hidden_state for embedding."""

    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
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
    if path.startswith(("http://", "https://")):
        parsed = urlparse(path)
        clean_path = parsed.path
    else:
        clean_path = path
    _, ext = os.path.splitext(clean_path.lower())
    return ext in image_extensions


def _is_video_input(video: Any) -> bool:
    # Keep minimal parity with official helper
    if isinstance(video, str):
        return True
    if isinstance(video, list) and len(video) > 0:
        first = video[0]
        if isinstance(first, Image.Image):
            return True
        if isinstance(first, str):
            return _is_image_path(first)
    return False


import importlib.util

def _has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


class Qwen3VLEmbedder:
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        attn_implementation: str = "sdpa",   # <- 新增：显式默认 sdpa
        **from_pretrained_kwargs,
    ):
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

        # ---- 关键：flash-attn2 不可用就自动降级 ----
        if attn_implementation == "flash_attention_2" and not _has_flash_attn():
            print("[WARN] flash_attn not installed, fallback attn_implementation=sdpa")
            attn_implementation = "sdpa"

        # 让 transformers 走正确的 attention backend
        from_pretrained_kwargs = dict(from_pretrained_kwargs)
        from_pretrained_kwargs["attn_implementation"] = attn_implementation
        from_pretrained_kwargs.setdefault("trust_remote_code", True)

        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path,
            **from_pretrained_kwargs,
        ).to(self.device)
        self.model.eval()

        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path,
            padding_side="right",
        )


    @staticmethod
    def _ensure_instruction_punct(instr: str) -> str:
        instr = instr.strip()
        if instr and not unicodedata.category(instr[-1]).startswith("P"):
            instr = instr + "."
        return instr

    def format_model_input(
        self,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[Union[List[Any], Any]] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if instruction:
            instruction = self._ensure_instruction_punct(instruction)
        else:
            instruction = self.default_instruction

        # normalize
        texts = []
        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, list):
            texts = text

        images = []
        if image is None:
            images = []
        elif isinstance(image, list):
            images = image
        else:
            images = [image]

        videos = []
        if video is None:
            videos = []
        elif _is_video_input(video):
            videos = [video]
        else:
            videos = list(video)

        content = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

        if not texts and not images and not videos:
            content.append({"type": "text", "text": "NULL"})
            return conversation

        # images
        for img in images:
            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = img if img.startswith(("http://", "https://")) else ("file://" + img)
            else:
                raise TypeError(f"Unrecognized image type: {type(img)}")
            content.append(
                {
                    "type": "image",
                    "image": image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )

        # texts
        for t in texts:
            content.append({"type": "text", "text": t})

        # video support omitted here because你的任务是“先只做图片库向量化”
        # 若后续要加视频，直接参考官方实现扩展即可。

        return conversation

    def _preprocess_inputs(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        texts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]

        # --- version-compatible call ---
        try:
            out = process_vision_info(conversations, image_patch_size=16)
        except TypeError:
            # older qwen_vl_utils: no image_patch_size arg
            out = process_vision_info(conversations)

        # handle return signatures (2-tuple or 3-tuple)
        images = None
        videos = None
        video_metadata = None
        video_kwargs = {}

        if isinstance(out, tuple):
            if len(out) == 2:
                images, videos = out
            elif len(out) == 3:
                images, videos, video_kwargs = out
            else:
                raise RuntimeError(f"process_vision_info returned unexpected tuple length: {len(out)}")
        else:
            raise RuntimeError(f"process_vision_info returned unexpected type: {type(out)}")

        inputs = self.processor(
            text=texts,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=True,
            return_tensors="pt",
            **video_kwargs,
        )
        return inputs


    @staticmethod
    def _pool_last_valid_token(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last valid token position per sample
        flipped = attention_mask.flip(dims=[1])
        last_one = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_one - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    @torch.no_grad()
    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        conversations = [
            self.format_model_input(
                text=ele.get("text"),
                image=ele.get("image"),
                instruction=ele.get("instruction"),
            )
            for ele in inputs
        ]
        model_inputs = self._preprocess_inputs(conversations)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        outputs = self.model(**model_inputs)
        emb = self._pool_last_valid_token(outputs.last_hidden_state, model_inputs["attention_mask"])
        if normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


# ---- Your embedding pipeline below ----

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=str, required=True, help="Input shard JSONL.")
    ap.add_argument("--out_dir", type=str, default="/home/DATA2/hwxdataset/emb_shards",
                    help="Output directory for embedding shards.")
    ap.add_argument("--model_path", type=str, required=True,
                    help="Local path of Qwen3-VL-Embedding checkpoint directory.")
    ap.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g., cuda:0.")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size per forward.")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"],
                    help="Model dtype.")
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"],
                    help="Attention implementation (if supported by your build).")
    ap.add_argument("--instruction", type=str,
                    default="Retrieve images relevant to the user's query.",
                    help="Instruction for embedding model.")
    ap.add_argument("--out_dim", type=int, default=4096,
                    help="Output embedding dim. If smaller than native, slice (MRL-style).")
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalize embeddings (recommended for cosine/IP).")
    ap.add_argument("--max_items", type=int, default=-1, help="Max items to process. -1 = all.")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if output files already exist.")
    ap.add_argument("--fail_fast", action="store_true", help="Stop on first model error.")
    return ap.parse_args()


def verify_image(path: str) -> Optional[str]:
    """Return error string if invalid, else None."""
    try:
        with Image.open(path) as im:
            im.verify()
        return None
    except Exception as e:
        return f"{type(e).__name__}:{e}"


def main():
    args = parse_args()
    shard_path = Path(args.shard)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = shard_path.stem
    emb_path = out_dir / f"{stem}.emb.npy"
    ids_path = out_dir / f"{stem}.ids.txt"
    bad_path = out_dir / f"{stem}.bad.tsv"

    if args.skip_existing and emb_path.exists() and ids_path.exists():
        print(f"[SKIP] {stem}: outputs exist at {out_dir}")
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # Build embedder (Transformers + local checkpoint)
    model = Qwen3VLEmbedder(
        model_name_or_path=args.model_path,
        device=args.device,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,  # <- 这里会自动降级
    )


    # Load records
    recs: List[Dict[str, Any]] = []
    with shard_path.open("r", encoding="utf-8") as rf:
        for i, line in enumerate(rf):
            if args.max_items != -1 and i >= args.max_items:
                break
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue

    if not recs:
        print(f"[WARN] empty shard: {shard_path}")
        return

    out_emb_list: List[np.ndarray] = []
    out_id_lines: List[str] = []

    with open(bad_path, "w", encoding="utf-8") as bad_fh:
        bad_fh.write("img_id\tpath\terror\n")

        def flush_batch(batch_items: List[Tuple[int, str]]):
            if not batch_items:
                return
            inputs = [{"image": p, "instruction": args.instruction} for (_, p) in batch_items]
            try:
                embs = model.process(inputs, normalize=args.normalize)  # torch.Tensor [B, D]
            except Exception as e:
                if args.fail_fast:
                    raise
                for img_id, p in batch_items:
                    bad_fh.write(f"{img_id}\t{p}\tMODEL_ERROR:{repr(e)}\n")
                return

            embs = embs.detach().to("cpu").float().numpy()  # float32
            if embs.ndim != 2:
                for img_id, p in batch_items:
                    bad_fh.write(f"{img_id}\t{p}\tBAD_EMB_SHAPE:{embs.shape}\n")
                return

            # slice dim if requested
            if args.out_dim and 0 < args.out_dim < embs.shape[1]:
                embs = embs[:, :args.out_dim]

            out_emb_list.append(embs.astype(np.float16, copy=False))
            for img_id, p in batch_items:
                out_id_lines.append(f"{img_id}\t{p}\n")

        batch: List[Tuple[int, str]] = []
        pbar = tqdm(recs, desc=f"Embedding {stem}", ncols=100)

        for rec in pbar:
            img_id = rec.get("img_id", -1)
            path = rec.get("path", "")
            if not path or not isinstance(path, str):
                bad_fh.write(f"{img_id}\t{path}\tNO_PATH\n")
                continue
            if not os.path.exists(path):
                bad_fh.write(f"{img_id}\t{path}\tNOT_FOUND\n")
                continue

            err = verify_image(path)
            if err is not None:
                bad_fh.write(f"{img_id}\t{path}\tBAD_IMAGE:{err}\n")
                continue

            batch.append((int(img_id), path))
            if len(batch) >= args.batch_size:
                flush_batch(batch)
                batch = []

        flush_batch(batch)

    if not out_emb_list:
        print(f"[WARN] no embeddings produced for {stem}. Check {bad_path}")
        with open(ids_path, "w", encoding="utf-8") as wf:
            wf.write("")
        np.save(emb_path, np.zeros((0, args.out_dim), dtype=np.float16))
        return

    embs_all = np.concatenate(out_emb_list, axis=0)
    np.save(emb_path, embs_all)

    with open(ids_path, "w", encoding="utf-8") as wf:
        wf.writelines(out_id_lines)

    print(f"[DONE] shard={shard_path}")
    print(f"[DONE] emb={emb_path} shape={embs_all.shape} dtype={embs_all.dtype}")
    print(f"[DONE] ids={ids_path} lines={len(out_id_lines)}")
    print(f"[DONE] bad={bad_path}")


if __name__ == "__main__":
    main()
