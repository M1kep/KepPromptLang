"""DSL-aware CLIP text encoders.

These subclass ComfyUI's `SDClipModel` and override `encode_token_weights` to:

  1. Walk our `List[List[SegOrAction]]` token structure.
  2. Resolve each `Action` into an embedding tensor (and optional position-embedding
     post-modifiers) via `Action.get_result(token_embedding)`.
  3. Concatenate per-batch embeddings into `[B, seq, hidden]`.
  4. Pass straight to `self.transformer(None, mask, embeds=..., num_tokens=...)`.

Position-embedding-modifying actions (`posScale`, `postPos`) are supported without
patching the transformer — see `_apply_pos_modifiers`.
"""

import dataclasses
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding

from comfy import sd1_clip, sdxl_clip

from .actions.base import Action, PostModifiers
from .actions.types import SegOrAction


class PromptLangSDClipModel(sd1_clip.SDClipModel):
    """Drop-in replacement for SDClipModel that consumes our segment/action token stream."""

    def encode_token_weights(self, prompt_segments: List[List[SegOrAction]]):  # type: ignore[override]
        device = self._resolve_device()
        embedding_module = self.transformer.get_input_embeddings()

        embeds, pos_modifiers_per_batch = _build_embeddings(prompt_segments, embedding_module, device)
        attention_mask, num_tokens = _build_attention_mask(
            prompt_segments, device, end_token=self.special_tokens.get("end")
        )

        embeds_for_transformer = _apply_pos_modifiers(
            embeds, pos_modifiers_per_batch, position_embedding=self._get_position_embedding()
        )

        attention_mask_model = attention_mask if self.enable_attention_masks else None

        outputs = self.transformer(
            None,
            attention_mask_model,
            embeds=embeds_for_transformer,
            num_tokens=num_tokens,
            intermediate_output=self.layer_idx if self.layer == "hidden" else None,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
            dtype=torch.float32,
        )

        z = outputs[0].float() if self.layer == "last" else outputs[1].float()

        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        return z, pooled_output

    def _resolve_device(self):
        if self.execution_device is not None:
            return self.execution_device
        return self.transformer.get_input_embeddings().weight.device

    def _get_position_embedding(self) -> Embedding:
        """Reach into ComfyUI's CLIPTextModel for the position-embedding table.

        Isolated so a ComfyUI internal refactor only needs one fix.
        """
        return self.transformer.text_model.embeddings.position_embedding


class PromptLangSDXLClipG(sdxl_clip.SDXLClipG, PromptLangSDClipModel):
    """SDXL's larger CLIP-G text encoder, with our DSL-aware encode_token_weights."""


def _build_embeddings(
    batches: List[List[SegOrAction]],
    embedding_module: Embedding,
    device,
) -> Tuple[Tensor, List[List[PostModifiers]]]:
    """Run actions/segments to produce per-batch embedding tensors and post-modifier lists."""
    per_batch_embeds: List[Tensor] = []
    per_batch_modifiers: List[List[PostModifiers]] = []

    for batch in batches:
        pieces: List[Tensor] = []
        modifiers: List[PostModifiers] = []
        token_idx = 0

        for seg_or_action in batch:
            if isinstance(seg_or_action, Action):
                result = seg_or_action.get_result(embedding_module)
                if isinstance(result, tuple):
                    tensor, post_mods = result
                    end_idx = token_idx + seg_or_action.token_length()
                    modifiers.append(dataclasses.replace(post_mods, start_idx=token_idx, end_idx=end_idx))
                else:
                    tensor = result
            else:
                tensor = seg_or_action.get_embeddings(embedding_module)
            pieces.append(tensor)
            token_idx += seg_or_action.token_length()

        # One cat+cast per batch instead of per-piece — avoids N kernels for N segments.
        per_batch_embeds.append(torch.cat(pieces, dim=1).to(device=device, dtype=torch.float32))
        per_batch_modifiers.append(modifiers)

    return torch.cat(per_batch_embeds, dim=0), per_batch_modifiers


def _build_attention_mask(
    batches: List[List[SegOrAction]],
    device,
    end_token,
) -> Tuple[Tensor, List[int]]:
    """Build a 1-where-real, 0-where-padded mask plus a per-batch real-token count.

    EOS is detected by scanning `PromptSegment` integer tokens; action-produced tokens
    are always treated as real content (they never contain the EOS marker).
    """
    masks = []
    num_tokens = []

    for batch in batches:
        mask: List[int] = []
        eos_seen = False
        for seg_or_action in batch:
            if isinstance(seg_or_action, Action):
                # All action-produced tokens are real content (no EOS inside).
                mask.extend([0] * seg_or_action.token_length() if eos_seen else [1] * seg_or_action.token_length())
                continue
            for tok in seg_or_action.tokens:
                if eos_seen:
                    mask.append(0)
                    continue
                mask.append(1)
                if end_token is not None and isinstance(tok, int) and tok == end_token:
                    eos_seen = True
        masks.append(mask)
        num_tokens.append(sum(mask))

    return torch.tensor(masks, device=device, dtype=torch.long), num_tokens


def _apply_pos_modifiers(
    embeds: Tensor,
    pos_modifiers_per_batch: List[List[PostModifiers]],
    position_embedding: Embedding,
) -> Tensor:
    """Pre-bake position-modifier deltas into `embeds` so the transformer's inline add yields the modified pos embedding.

    `comfy.clip_model.CLIPTextModel_.forward` does `x = embeds + position_embedding.weight[:seq]`
    whenever `embeds` is passed in. To end up with a *modified* position embedding for a slice,
    we add `(modified - default)` here so the transformer's add-back (`+ default`) nets to
    `+ modified`. Not an assignment — `embeds` already carries the token embeddings.
    """
    if not any(pos_modifiers_per_batch):
        return embeds

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

            out[batch_idx, mod.start_idx:mod.end_idx] += modified_slice - default_slice

    return out


class PromptLangSD1ClipModel(sd1_clip.SD1ClipModel):
    """SD1.x wrapper using our DSL-aware single CLIP-L encoder."""

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
    """SDXL wrapper using our DSL-aware CLIP-L and CLIP-G encoders."""

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
