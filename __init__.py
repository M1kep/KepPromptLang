from .nodes import ClipInjectedCheckpointLoader, FunCLIPTextEncode, BuildGif

NODE_CLASS_MAPPINGS = {
    "ClipInjectedCheckpointLoader": ClipInjectedCheckpointLoader,
    "FunCLIPTextEncode": FunCLIPTextEncode,
    "Build Gif": BuildGif
}
#
# EXTENSION_NAME = "ComfyLiterals"
# symlink_web_dir("js", EXTENSION_NAME)
