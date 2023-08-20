from .nodes import (
    FunCLIPTextEncode,
    BuildGif,
    SpecialClipLoader,
)

NODE_CLASS_MAPPINGS = {
    "FunCLIPTextEncode": FunCLIPTextEncode,
    "Build Gif": BuildGif,
    "Special CLIP Loader": SpecialClipLoader
}
