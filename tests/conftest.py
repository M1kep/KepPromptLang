"""Test setup that runs before any tests are collected.

Two things make this tricky:
  1. The project's runtime imports use ComfyUI (`comfy.sd1_clip`), which we don't want to require for unit tests.
  2. The package is normally installed under `custom_nodes/KepPromptLang/`, so we register `KepPromptLang` as a package alias.
"""

import os
import sys
import types

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(REPO_ROOT))


def _install_runtime_stubs():
    """Stub out runtime deps (numpy/PIL/comfy/folder_paths) so test imports of the package work.

    Tests don't exercise the ComfyUI nodes; they only need the parser and action math layers.
    """
    # Only stub modules that aren't actually installed; real numpy/torch must take precedence.
    if "PIL" not in sys.modules:
        try:
            import PIL  # noqa: F401
        except ImportError:
            pil = types.ModuleType("PIL")
            pil.Image = types.ModuleType("PIL.Image")
            sys.modules["PIL"] = pil
            sys.modules["PIL.Image"] = pil.Image

    if "numpy" not in sys.modules:
        try:
            import numpy  # noqa: F401
        except ImportError:
            sys.modules["numpy"] = types.ModuleType("numpy")

    if "folder_paths" not in sys.modules:
        sys.modules["folder_paths"] = types.ModuleType("folder_paths")

    if "comfy" in sys.modules:
        return

    comfy = types.ModuleType("comfy")
    comfy_sd = types.ModuleType("comfy.sd")
    comfy_sd.CLIP = type("CLIP", (), {})
    comfy_supported = types.ModuleType("comfy.supported_models_base")
    comfy_supported.ClipTarget = type("ClipTarget", (), {})
    sdxl_clip = types.ModuleType("comfy.sdxl_clip")
    sdxl_clip.SDXLClipModel = type("SDXLClipModel", (), {})
    sdxl_clip.SDXLClipG = type("SDXLClipG", (), {})
    sd1_clip = types.ModuleType("comfy.sd1_clip")

    class SDTokenizer:  # minimal stand-in
        embedding_identifier = "embedding:"

        def __init__(self, *args, **kwargs):
            self.embedding_directory = None
            self.embedding_size = 768
            self.start_token = 49406
            self.end_token = 49407
            self.pad_with_end = True
            self.max_length = 77
            self.tokenizer = _FakeTokenizer()

        def _try_get_embedding(self, name):
            return None, ""

    class SD1Tokenizer:
        def __init__(self, *args, **kwargs):
            pass

    sd1_clip.SDTokenizer = SDTokenizer
    sd1_clip.SD1Tokenizer = SD1Tokenizer
    sd1_clip.SDClipModel = type("SDClipModel", (), {})
    sd1_clip.SD1ClipModel = type("SD1ClipModel", (), {})
    comfy.sd1_clip = sd1_clip
    comfy.sd = comfy_sd
    comfy.sdxl_clip = sdxl_clip
    comfy.supported_models_base = comfy_supported
    sys.modules["comfy"] = comfy
    sys.modules["comfy.sd1_clip"] = sd1_clip
    sys.modules["comfy.sdxl_clip"] = sdxl_clip
    sys.modules["comfy.sd"] = comfy_sd
    sys.modules["comfy.supported_models_base"] = comfy_supported


class _FakeTokenizer:
    """Tokenize each whitespace-separated word into a single deterministic int id."""

    def __call__(self, word):
        # Deterministic: sum of character codepoints, modulo a small range. SOT/EOT bracketing.
        token = (sum(ord(c) for c in word) % 49000) + 100
        return {"input_ids": [49406, token, 49407]}


_install_runtime_stubs()


@pytest.fixture
def tokenizer():
    from comfy.sd1_clip import SDTokenizer

    return SDTokenizer()
