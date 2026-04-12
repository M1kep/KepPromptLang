import os
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
from PIL import Image

import comfy.sd
import folder_paths
from comfy.supported_models_base import ClipTarget

from .lib.clip_model import PromptLangSD1ClipModel, PromptLangSDXLClipModel
from .lib.tokenizer import PromptLangSD1Tokenizer, PromptLangSDXLTokenizer


class SpecialClipLoader:
    """Wraps a loaded CLIP with our DSL-aware tokenizer + text encoder."""

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore[no-untyped-def]
        return {
            "required": {
                "source_clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "conditioning"

    @staticmethod
    def load_clip(source_clip: comfy.sd.CLIP) -> Tuple[comfy.sd.CLIP]:
        is_sdxl = hasattr(source_clip.cond_stage_model, "clip_g") and hasattr(source_clip.cond_stage_model, "clip_l")
        embedding_directory = source_clip.tokenizer.clip_l.embedding_directory

        if is_sdxl:
            target = ClipTarget(PromptLangSDXLTokenizer, PromptLangSDXLClipModel)
        else:
            target = ClipTarget(PromptLangSD1Tokenizer, PromptLangSD1ClipModel)

        new_clip = comfy.sd.CLIP(target=target, embedding_directory=embedding_directory)

        if is_sdxl:
            new_clip.cond_stage_model.clip_l.transformer.load_state_dict(
                source_clip.cond_stage_model.clip_l.transformer.state_dict()
            )
            new_clip.cond_stage_model.clip_g.transformer.load_state_dict(
                source_clip.cond_stage_model.clip_g.transformer.state_dict()
            )
        else:
            new_clip.cond_stage_model.clip_l.transformer.load_state_dict(
                source_clip.cond_stage_model.clip_l.transformer.state_dict()
            )

        return (new_clip,)


def tensor2img(tensor_img) -> Image.Image:
    arr = (255.0 * tensor_img.cpu().numpy()).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class BuildGif:
    """Builds an animated webp from a list of image batches.

    Two output modes:
      - "Big Grid": tiles batches across the X axis and chunks across the Y axis,
        producing a single animated webp where each frame is the next image in a chunk.
      - "One Per Split": one animation per (split, batch_index) combination.
    """

    def __init__(self) -> None:
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore[no-untyped-def]
        return {
            "required": {
                "images": ("IMAGE",),
                "split_every": ("INT", {"default": -1}),
                "frame_duration": ("INT", {"default": 125}),
                "output_mode": (["One Per Split", "Big Grid"], {"default": "Big Grid"}),
            }
        }

    RETURN_TYPES = ()
    INPUT_IS_LIST = True
    FUNCTION = "build_gif"
    OUTPUT_NODE = True
    CATEGORY = "List Stuff"

    def build_gif(
        self,
        images: List[Any],
        split_every: List[int],
        frame_duration: List[int],
        output_mode: List[str],
    ):
        if len(split_every) > 1:
            raise ValueError("List input for split_every is not supported.")
        if len(output_mode) > 1:
            raise ValueError("List input for output_mode is not supported.")
        if len(frame_duration) > 1:
            raise ValueError("List input for frame_duration is not supported.")

        mode = output_mode[0]
        duration = frame_duration[0]
        split_requested = split_every[0]

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix="Gif", output_dir=self.output_dir, image_width=0, image_height=0
        )

        batch_size = images[0].size()[0]
        # split_every=-1 means "don't split": one chunk containing everything.
        if split_requested == -1:
            split_chunks = 1
            chunk_len = len(images)
        else:
            chunk_len = split_requested
            split_chunks = len(images) // chunk_len

        chunked_batches = [
            images[chunk_len * i : chunk_len * (i + 1)]
            for i in range(split_chunks)
        ]

        results = []
        ctx = _SaveContext(
            images=images,
            chunked_batches=chunked_batches,
            chunk_len=chunk_len,
            batch_size=batch_size,
            split_chunks=split_chunks,
            full_output_folder=full_output_folder,
            filename=filename,
            counter=counter,
            subfolder=subfolder,
            duration=duration,
        )

        if mode == "Big Grid":
            results.append(self._save_big_grid(ctx))
        elif mode == "One Per Split":
            results.extend(self._save_one_per_split(ctx))
        return {"ui": {"images": results}}

    def _save_big_grid(self, ctx):
        img_shape = ctx.images[0][0].shape
        frames = []
        for idx_in_chunk in range(ctx.chunk_len):
            img_frame = Image.new(
                "RGB", size=(ctx.batch_size * img_shape[0], ctx.split_chunks * img_shape[1])
            )
            for split_idx in range(ctx.split_chunks):
                for batch_idx, img_tensor in enumerate(ctx.chunked_batches[split_idx][idx_in_chunk]):
                    img_frame.paste(
                        tensor2img(img_tensor),
                        (batch_idx * img_shape[0], split_idx * img_shape[1]),
                    )
            frames.append(img_frame)

        file = f"{ctx.filename}_{ctx.counter:05}_"
        save_path = os.path.join(ctx.full_output_folder, file)
        frames[0].save(
            f"{save_path}.webp",
            lossless=True,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=ctx.duration,
            loop=0,
        )
        return {"filename": f"{file}.webp", "subfolder": ctx.subfolder, "type": "output"}

    def _save_one_per_split(self, ctx):
        results = []
        counter = ctx.counter
        for split_idx in range(ctx.split_chunks):
            split_start = ctx.chunk_len * split_idx
            split_end = ctx.chunk_len * (split_idx + 1)
            for batch_idx in range(ctx.batch_size):
                file = f"{ctx.filename}_{counter:05}_"
                save_path = os.path.join(ctx.full_output_folder, file)
                counter += 1
                tensor2img(ctx.images[split_start][batch_idx]).save(
                    f"{save_path}.webp",
                    save_all=True,
                    append_images=[
                        tensor2img(nested[batch_idx])
                        for nested in ctx.images[split_start + 1 : split_end]
                    ],
                    optimize=False,
                    duration=ctx.duration,
                    loop=0,
                )
                results.append({"filename": f"{file}.webp", "subfolder": ctx.subfolder, "type": "output"})
        return results


@dataclass
class _SaveContext:
    images: Any
    chunked_batches: Any
    chunk_len: int
    batch_size: int
    split_chunks: int
    full_output_folder: str
    filename: str
    counter: int
    subfolder: str
    duration: int
