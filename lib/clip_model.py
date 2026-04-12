"""DSL-aware CLIP text encoders.

The tokenizer emits ComfyUI's native `(token, weight)` format with one twist:
a `token` can also be a lazily-evaluated `Action`. We resolve those to tensors
here in `process_tokens` (where the embedding module is available) and delegate
everything else — embedding lookup, mask building, splice — to the stock
`SDClipModel.process_tokens`.

`posScale` / `postPos` actions return a `PostModifiers` alongside their tensor.
ComfyUI's `CLIPTextModel_.forward` adds the position embedding inline whenever
`embeds` is supplied, so we pre-bake `(modified - default)` into `embeds` such
that the transformer's add nets to `+ modified`.
"""

import dataclasses
from typing import List

import torch
from torch import Tensor
from torch.nn import Embedding

from comfy import sd1_clip, sdxl_clip

from .actions.base import Action, PostModifiers


class PromptLangSDClipModel(sd1_clip.SDClipModel):
    def process_tokens(self, tokens, device):  # type: ignore[override]
        embedding_module = self.transformer.get_input_embeddings()

        resolved: List[list] = []
        pos_modifiers_per_batch: List[List[PostModifiers]] = []

        for batch in tokens:
            row: list = []
            modifiers: List[PostModifiers] = []
            position = 0
            for entry in batch:
                if isinstance(entry, Action):
                    length = entry.token_length()
                    result = entry.get_result(embedding_module)
                    if isinstance(result, tuple):
                        tensor, mods = result
                        modifiers.append(
                            dataclasses.replace(mods, start_idx=position, end_idx=position + length)
                        )
                    else:
                        tensor = result
                    row.append(tensor)
                    position += length
                else:
                    row.append(entry)
                    position += 1
            resolved.append(row)
            pos_modifiers_per_batch.append(modifiers)

        embeds, attention_mask, num_tokens, embeds_info = super().process_tokens(resolved, device)

        if any(pos_modifiers_per_batch):
            embeds = _apply_pos_modifiers(
                embeds, pos_modifiers_per_batch, self._get_position_embedding()
            )

        return embeds, attention_mask, num_tokens, embeds_info

    def _get_position_embedding(self) -> Embedding:
        """Isolated so a ComfyUI internal layout change only needs one fix."""
        return self.transformer.text_model.embeddings.position_embedding


class PromptLangSDXLClipG(sdxl_clip.SDXLClipG, PromptLangSDClipModel):
    """SDXL's larger CLIP-G text encoder, with our DSL-aware process_tokens."""


def _apply_pos_modifiers(
    embeds: Tensor,
    pos_modifiers_per_batch: List[List[PostModifiers]],
    position_embedding: Embedding,
) -> Tensor:
    seq_len = embeds.shape[1]
    pos_weights = position_embedding.weight[:seq_len].to(device=embeds.device, dtype=embeds.dtype)

    out = embeds.clone()
    for batch_idx, modifiers in enumerate(pos_modifiers_per_batch):
        for mod in modifiers:
            default_slice = pos_weights[mod.start_idx:mod.end_idx]

            if mod.bypass_pos_embed:
                modified_slice = torch.zeros_like(default_slice)
            elif mod.position_embed_scale is not None:
                modified_slice = default_slice * float(mod.position_embed_scale)
            else:
                continue

            # The transformer will add `default_slice` back; net effect is `+ modified_slice`.
            out[batch_idx, mod.start_idx:mod.end_idx] += modified_slice - default_slice

    return out


class PromptLangSD1ClipModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options=None, **kwargs):
        super().__init__(
            device=device,
            dtype=dtype,
            model_options=model_options or {},
            clip_name="l",
            clip_model=PromptLangSDClipModel,
            **kwargs,
        )


class PromptLangSDXLClipModel(sdxl_clip.SDXLClipModel):
    def __init__(self, device="cpu", dtype=None, model_options=None) -> None:
        torch.nn.Module.__init__(self)
        opts = model_options or {}
        self.clip_l = PromptLangSDClipModel(
            layer="hidden",
            layer_idx=-2,
            device=device,
            dtype=dtype,
            layer_norm_hidden_state=False,
            model_options=opts,
        )
        self.clip_g = PromptLangSDXLClipG(device=device, dtype=dtype, model_options=opts)
        self.dtypes = {dtype} if dtype is not None else set()
