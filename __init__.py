from .nodes import BuildGif, SpecialClipLoader

NODE_CLASS_MAPPINGS = {
    "Build Gif": BuildGif,
    "Special CLIP Loader": SpecialClipLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Build Gif": "Build GIF (KepPromptLang)",
    "Special CLIP Loader": "Special CLIP Loader (KepPromptLang)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
