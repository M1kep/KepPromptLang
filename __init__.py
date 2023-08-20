from .nodes import (
    FunCLIPTextEncode,
    BuildGif,
    ClipPatcher,
    SpecialClipLoader,
)

NODE_CLASS_MAPPINGS = {
    "FunCLIPTextEncode": FunCLIPTextEncode,
    "Build Gif": BuildGif,
    "Clip Patcher": ClipPatcher,
    "Special CLIP Loader": SpecialClipLoader
}
