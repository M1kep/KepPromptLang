from .nodes import BuildGif, PromptLangInspect, SpecialClipLoader

NODE_CLASS_MAPPINGS = {
    "Build Gif": BuildGif,
    "Special CLIP Loader": SpecialClipLoader,
    "PromptLang Inspect": PromptLangInspect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Build Gif": "Build GIF (KepPromptLang)",
    "Special CLIP Loader": "Special CLIP Loader (KepPromptLang)",
    "PromptLang Inspect": "PromptLang Inspect",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
