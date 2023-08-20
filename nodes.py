import random

import numpy as np
import torch
from PIL import Image

import folder_paths
import comfy.sd
from comfy import model_management
from custom_nodes.ClipStuff.lib.clip_model import SD1FunClipModel, FunCLIP
from custom_nodes.ClipStuff.lib.tokenizer import MyTokenizer


class ClipInjectedCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"config_name": (folder_paths.get_filename_list("configs"),),
                             "ckpt_name": (folder_paths.get_filename_list("checkpoints"),)}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "advanced/loaders"

    def load_checkpoint(self, config_name, ckpt_name, output_vae=True, output_clip=True):
        config_path = folder_paths.get_full_path("configs", config_name)
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        return comfy.sd.load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=True,
                                        embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                        clip_model=SD1FunClipModel,
                                        clip_class=FunCLIP,
                                        clip_tokenizer=MyTokenizer)


class FunCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"multiline": True}), "clip": ("CLIP",),
            # "slerp_power": ("FLOAT", {"min": 0.0, "max": 1.0}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "conditioning"

    def encode(self, clip, text):
        ret = []
        for prompt in text.split("\n"):
            if prompt.strip() == "":
                continue
            tokens = clip.tokenizer.tokenize_with_weights(text, return_word_ids=False,)
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


#
# def buildGif(processing_res, path='./outputs/gif/'):
#     width = processing_res.width
#     height = processing_res.height
#
#     x_batch[0].save(f"{path}{speed}_{processing_res.seed}_batch_{y}.gif", save_all=True, append_images=x_batch[1:],
#                     optimize=False, duration=speed, loop=0)
#
#     count_x = int(processing_res.images[0].width / width)
#     count_y = int(processing_res.images[0].height / height)
#
#     gif_interval = sharedObj.Config['gif_interval']
#     if gif_interval.find(",") > 0:
#         speeds = list(map(int, gif_interval.split(",")))
#     else:
#         speeds = [int(gif_interval)]
#
#     gif_axis = sharedObj.Config['gif_axis']
#     # ax1 = count_y if gif_axis == "X" else count_x
#     # ax2 = count_x if gif_axis == "X" else count_y
#     if gif_axis == "X":
#         for y in range(0, count_y):
#             x_batch = []
#             for x in range(0, count_x):
#                 bbox = (x * width, y * height, (x + 1) * width, (y + 1) * height)
#                 print(bbox)
#                 x_batch.append(processing_res.images[0].crop(bbox))
#                 # images.save_image(processed.images[g], p.outpath_grids, "xyz_grid"
#                 # working_slice.show()
#             if sharedConfig.Config.get('gif_boomerang', False):
#                 boomerang_in_place(x_batch)
#             for speed in speeds:
#
#     else:
#         print("Y!")
#         for x in range(0, count_x):
#             y_batch = []
#             for y in range(0, count_y):
#                 bbox = (x * width, y * height, (x + 1) * width, (y + 1) * height)
#                 print(bbox)
#                 y_batch.append(processing_res.images[0].crop(bbox))
#                 # images.save_image(processed.images[g], p.outpath_grids, "xyz_grid"
#                 # working_slice.show()
#             if sharedConfig.Config.get('gif_boomerang', False):
#                 boomerang_in_place(y_batch)
#             for speed in speeds:
#                 y_batch[0].save(f"{path}{speed}_{processing_res.seed}_batch_{x}.gif", save_all=True, append_images=y_batch[1:], optimize=False, duration=speed, loop=0)


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
            },
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Gifs",)
    INPUT_IS_LIST = True
    FUNCTION = "build_gif"
    OUTPUT_IS_LIST = (True,)
    # OUTPUT_NODE = False

    CATEGORY = "List Stuff"

    def build_gif(self, images: list):
        print('Build GIF called!')
        print(f"{type(images)}")

        batch_size = images[0].size()[0]
        cell_width = images[0].size()[1]
        cell_height = images[0].size()[2]
        # x_batch[0].save(f"{path}{speed}_{processing_res.seed}_batch_{y}.gif", save_all=True, append_images=x_batch[1:],
        #                 optimize=False, duration=speed, loop=0)

        out = []

        for batch_idx in range(batch_size):
            save_path = f"{folder_paths.get_output_directory()}/-{batch_idx}-{random.randint(1, 100)}"
            print(save_path)
            # tensor2img(images[0][batch_idx]).save(
            #     f"{save_path}.gif",
            #     save_all=True,
            #     append_images=[
            #         tensor2img(nested_batch[batch_idx]) for nested_batch in images[1:]
            #     ],
            #     optimize=False,
            #     duration=100,
            #     loop=0
            # )
            tensor2img(images[0][batch_idx]).save(
                f"{save_path}.webp",
                save_all=True,
                append_images=[
                    tensor2img(nested_batch[batch_idx]) for nested_batch in images[1:]
                ],
                optimize=False,
                duration=250,
                loop=0
            )


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
