from .nodes import (
    BuildGif,
    SpecialClipLoader,
    MonacoPrompt,
)

NODE_CLASS_MAPPINGS = {
    "Build Gif": BuildGif,
    "Special CLIP Loader": SpecialClipLoader,
    "Monaco Prompt": MonacoPrompt,
}

WEB_DIRECTORY = ("./web/dist", ["app.bundle.js"])
