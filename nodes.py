import random

import numpy as np
from PIL import Image

import folder_paths
import comfy.sd
import comfy.ops
from custom_nodes.ClipStuff.lib.clip_model import SD1FunClipModel

from custom_nodes.ClipStuff.lib.tokenizer import MyTokenizer
class EmptyClass:
    pass

class SpecialClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "source_clip": ("CLIP",),
        }}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    OUTPUT_IS_LIST = (False,)
    CATEGORY = "conditioning"

    def load_clip(self, source_clip):
        clip_target = EmptyClass()
        clip_target.params = {}
        clip_target.clip = SD1FunClipModel
        clip_target.tokenizer = MyTokenizer

        # TODO: Extract embedding directory from source_clip
        clip = comfy.sd.CLIP(clip_target, embedding_directory=None)
        comfy.sd.load_clip_weights(clip.cond_stage_model, source_clip.cond_stage_model.state_dict())
        return (clip,)

class FunCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"multiline": True}), "clip": ("CLIP",),
            "nudge_start": ("INT", {}),
            "nudge_end": ("INT", {})
            # "slerp_power": ("FLOAT", {"min": 0.0, "max": 1.0}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "conditioning"

    def encode(self, clip, text, nudge_start, nudge_end):
        ret = []
        for prompt in text.split("\n"):
            if prompt.strip() == "":
                continue
            tokens = clip.tokenizer.tokenize_with_weights(text, return_word_ids=False, nudge_start=nudge_start, nudge_end=nudge_end)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True, position_ids=[0] * 77)
            cond = [[cond, {"pooled_output": pooled}]]
            ret.append(cond)
        # if clip.layer_idx is not None:
        #     clip.cond_stage_model.clip_layer(clip.layer_idx)
        # else:
        #     clip.cond_stage_model.reset_clip_layer()
        #
        # model_management.load_model_gpu(clip.patcher)
        # position_ids = [0] * 77
        # cond, pooled = clip.cond_stage_model.encode_token_weights(tokens, position_ids=position_ids)
        # # return cond, pooled

        # return ([[cond, {"pooled_output": pooled}]],)
        return (ret,)

def tensor2img(tensor_img):
    i = 255. * tensor_img.cpu().numpy()
    i_np_arr = np.clip(i, 0, 255, out=i).astype(np.uint8, copy=False)
    return Image.fromarray(i_np_arr)


class BuildGif:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "split_every": ("INT", {"default": -1})
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

    def build_gif(self, images: list, split_every: list[int]):
        print('Build GIF called!')
        print(f"{type(images)}")

        if len(split_every) > 1:
            raise Exception("List input for split every is not supported.")
        split_every = split_every[0]

        batch_size = images[0].size()[0]
        if split_every == -1:
            split_chunks = 1
            split_every = len(images)
        else:
            split_chunks = int(len(images)/split_every)
        cell_width = images[0].size()[1]
        cell_height = images[0].size()[2]
        # x_batch[0].save(f"{path}{speed}_{processing_res.seed}_batch_{y}.gif", save_all=True, append_images=x_batch[1:],
        #                 optimize=False, duration=speed, loop=0)

        out = []

        num_wide = batch_size
        num_tall = split_chunks

        chunked_batches = [images[split_every * chunk_idx:split_every * (chunk_idx + 1)] for chunk_idx in range(split_chunks)]



        frames = []
        # For every image in gif
        for idx_in_chunk in range(split_every):
            img_shape = images[0][0].shape
            img_frame = Image.new('RGB', size=(num_wide * img_shape[0], num_tall * img_shape[1]))
            # For every chunk of images
            for split_idx in range(split_chunks):
                img_chunk = chunked_batches[split_idx]
                for batch_idx, img_tensor in enumerate(img_chunk[idx_in_chunk]):
                    img = tensor2img(img_tensor)
                    img_frame.paste(img, (batch_idx * img_shape[0], split_idx * img_shape[1]))
            frames.append(img_frame)

        save_path = f"{folder_paths.get_output_directory()}/{random.randint(1, 100)}"
        frames[0].save(
                    f"{save_path}.webp",
                    # quality=100,
                    # method=6,
                    lossless=True,
                    save_all=True,
                    append_images=frames[1:],
                    optimize=False,
                    duration=125,
                    loop=0
        )
        # for split_idx in range(int(split_chunks)):
        #     split_start = split_every * split_idx
        #     split_end = split_every * (split_idx + 1)
        #     for batch_idx in range(batch_size):
        #         save_path = f"{folder_paths.get_output_directory()}/-{batch_idx}-{random.randint(1, 100)}"
        #         print(save_path)
        #         tensor2img(images[split_start][batch_idx]).save(
        #             f"{save_path}.webp",
        #             save_all=True,
        #             append_images=[
        #                 tensor2img(nested_batch[batch_idx]) for nested_batch in images[split_start + 1:split_end]
        #             ],
        #             optimize=False,
        #             duration=125,
        #             loop=0
        #         )



                # tensor2img(images[0][batch_idx]).save(
                #     f"{save_path}.webp",
                #     save_all=True,
                #     append_images=[
                #         tensor2img(nested_batch[batch_idx]) for nested_batch in images[1:]
                #     ],
                #     optimize=False,
                #     duration=125,
                #     loop=0
                # )


        # for idx, img_batch in enumerate(images):
        #     for batch_idx, img in enumerate(img_batch):
        #         print("Stuff")
        #         img = tensor2img(img)
        #         pil_img = np.array(img.convert("RGB")).astype(np.float32, copy=False) / 255
        #         out.append(torch.from_numpy(pil_img).unsqueeze(0))


        #     box = (x * cell_width + margin + row_label_size, y * cell_height + margin + column_label_size)
        #     print(f"Box: {box}")
        #     print(f"Image: {type(img)}")
        #     grid_image.paste(img, box)
        #
        #     if y == 0:
        #         draw.text((box[0] + cell_width / 2, box[1] - column_label_size), str(Y_Labels[x]), fill='white',
        #                   font=font)
        #     if x == 0:
        #         draw.text((box[0] - row_label_size, box[1] + cell_width / 2), str(X_Labels[y]), fill='white', font=font)
        #
        # np_grid_image = np.array(grid_image.convert("RGB")).astype(np.float32, copy=False) / 255
        # torch_image = torch.from_numpy(np_grid_image)[None,]
        # np_grid_image = None
        # print(f"GridImage Shape: {grid_image.}")
        return (out,)
