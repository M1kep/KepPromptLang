import random
from typing import List

import numpy as np
from PIL import Image

import folder_paths
import comfy.sd
import comfy.ops
from custom_nodes.KepPromptLang.lib.clip_model import PromptLangClipModel

from custom_nodes.KepPromptLang.lib.tokenizer import PromptLangTokenizer


class EmptyClass:
    pass


class SpecialClipLoader:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "source_clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    OUTPUT_IS_LIST = (False,)
    CATEGORY = "conditioning"

    @staticmethod
    def load_clip(source_clip: comfy.sd.CLIP) -> tuple[comfy.sd.CLIP]:
        clip_target = EmptyClass()
        clip_target.params = {}
        clip_target.clip = PromptLangClipModel
        clip_target.tokenizer = PromptLangTokenizer

        clip = comfy.sd.CLIP(clip_target, embedding_directory=source_clip.tokenizer.embedding_directory)
        comfy.sd.load_clip_weights(
            clip.cond_stage_model, source_clip.cond_stage_model.state_dict()
        )
        return (clip,)


def tensor2img(tensor_img) -> Image.Image:
    i = 255.0 * tensor_img.cpu().numpy()
    i_np_arr = np.clip(i, 0, 255, out=i).astype(np.uint8, copy=False)
    return Image.fromarray(i_np_arr)


class BuildGif:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "images": ("IMAGE",),
                "split_every": ("INT", {"default": -1}),
                "output_mode": (
                    ["One Per Split", "Big Grid"],
                    {"default": "Big Grid"},
                ),
            }
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Gifs",)
    INPUT_IS_LIST = True
    FUNCTION = "build_gif"
    OUTPUT_IS_LIST = (True,)
    # OUTPUT_NODE = False

    CATEGORY = "List Stuff"

    @staticmethod
    def build_gif(images: list, split_every: List[int], output_mode: str):
        print("Build GIF called!")
        print(f"{type(images)}")

        if len(split_every) > 1:
            raise Exception("List input for split every is not supported.")

        split_every_val = split_every[0]
        batch_size = images[0].size()[0]
        if split_every_val == -1:
            split_chunks = 1
            split_every_val = len(images)
        else:
            split_chunks = int(len(images) / split_every_val)

        out = []

        num_wide = batch_size
        num_tall = split_chunks

        chunked_batches = [
            images[split_every_val * chunk_idx : split_every_val * (chunk_idx + 1)]
            for chunk_idx in range(split_chunks)
        ]

        frames = []

        if output_mode == "Big Grid":
            # For every image in gif
            for idx_in_chunk in range(split_every_val):
                img_shape = images[0][0].shape
                img_frame = Image.new(
                    "RGB", size=(num_wide * img_shape[0], num_tall * img_shape[1])
                )
                # For every chunk of images
                for split_idx in range(split_chunks):
                    img_chunk = chunked_batches[split_idx]
                    for batch_idx, img_tensor in enumerate(img_chunk[idx_in_chunk]):
                        img = tensor2img(img_tensor)
                        img_frame.paste(
                            img, (batch_idx * img_shape[0], split_idx * img_shape[1])
                        )
                frames.append(img_frame)

            save_path = (
                f"{folder_paths.get_output_directory()}/{random.randint(1, 100)}"
            )
            frames[0].save(
                f"{save_path}.webp",
                # quality=100,
                # method=6,
                lossless=True,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=125,
                loop=0,
            )
        elif output_mode == "One Per Split":
            for split_idx in range(int(split_chunks)):
                split_start = split_every_val * split_idx
                split_end = split_every_val * (split_idx + 1)
                for batch_idx in range(batch_size):
                    save_path = f"{folder_paths.get_output_directory()}/-{batch_idx}-{random.randint(1, 100)}"
                    print(save_path)
                    tensor2img(images[split_start][batch_idx]).save(
                        f"{save_path}.webp",
                        save_all=True,
                        append_images=[
                            tensor2img(nested_batch[batch_idx])
                            for nested_batch in images[split_start + 1 : split_end]
                        ],
                        optimize=False,
                        duration=125,
                        loop=0,
                    )
        return (out,)
