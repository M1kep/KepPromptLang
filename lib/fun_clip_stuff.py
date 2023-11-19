from typing import Optional, Tuple, Union, List, TypedDict, TYPE_CHECKING
from importlib.metadata import version as import_version
from packaging import version

import torch
from transformers import CLIPTextConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    CLIPTextEmbeddings,
    CLIPTextTransformer,
    CLIPTextModel,
)

from custom_nodes.KepPromptLang.lib.action.base import Action
from custom_nodes.KepPromptLang.lib.actions.types import SegOrAction

if TYPE_CHECKING:
    from torch import Tensor
    from custom_nodes.KepPromptLang.lib.action.base import PostModifiers


def slerp(val, low, high):
    low = low.unsqueeze(0)
    high = high.unsqueeze(0)
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


class PosModifier(TypedDict):
    """
    A dictionary of post modifiers for an action result.
    """

    position_embed_scale: Union[float]
    start_idx: Union[int]
    end_idx: Union[int]


class PromptLangCLIPTextEmbeddings(CLIPTextEmbeddings):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)

    def forward(
            self,
            input_dicts: Optional[List[List[SegOrAction]]] = None,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if input_dicts is None:
            raise ValueError("You have to specify input_dicts")

        batches: List[List[Tensor | Tuple[Tensor, PostModifiers] | Action]] = []
        pos_modifiers: List[List[PosModifier]] = []
        for batch_idx, batch in enumerate(input_dicts):
            results = []
            batch_pos_modifiers = []
            token_idx = 0
            for seg_or_action in batch:
                if isinstance(seg_or_action, Action):
                    action_result: Union[
                        Tensor, Tuple[Tensor, PostModifiers]
                    ] = seg_or_action.get_result(self.token_embedding)
                    if isinstance(action_result, tuple):
                        result, post_modifiers = action_result
                        if post_modifiers.get("position_embed_scale", None) is not None:
                            post_modifiers["start_idx"] = token_idx
                            post_modifiers["end_idx"] = (
                                token_idx + seg_or_action.token_length()
                            )

                        if post_modifiers.get("bypass_pos_embed", False):
                            post_modifiers["start_idx"] = token_idx
                            post_modifiers["end_idx"] = (
                                token_idx + seg_or_action.token_length()
                            )
                        batch_pos_modifiers.append(post_modifiers)
                    else:
                        result = action_result
                else:
                    result = seg_or_action.get_embeddings(self.token_embedding)
                results.append(result)
                token_idx += seg_or_action.token_length()
            batches.append(results)
            pos_modifiers.append(batch_pos_modifiers)

        seq_length = batches[0][0].shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        embeds = []
        for batch in batches:
            if len(batch) == 1:
                embeds.append(batch[0])
            else:
                embeds.append(torch.cat(batch, dim=-2))

        # Iterate over the batches and apply the pos modifiers to the position embeddings then add them to the embeddings
        for idx, batch_pos_modifiers in enumerate(pos_modifiers):
            position_embeddings = self.position_embedding(position_ids)
            if len(batch_pos_modifiers) > 0:
                print(f"Found {len(batch_pos_modifiers)} pos modifiers for batch {idx}")
                # Apply each pos modifier to the position embeddings at the specified indices
                for post_modifier in batch_pos_modifiers:
                    if post_modifier.get("bypass_pos_embed", False):
                        position_embeddings[
                            0, post_modifier["start_idx"] : post_modifier["end_idx"]
                        ] = 0
                    elif post_modifier["position_embed_scale"] is not None:
                        position_embeddings[
                            0, post_modifier["start_idx"] : post_modifier["end_idx"]
                        ] *= post_modifier["position_embed_scale"]
                    else:
                        raise ValueError(
                            "Pos modifier must have a scale or bypass_pos_embed"
                        )
            # Add the possibly modified position embeddings to the embeddings
            embeds[idx] = embeds[idx] + position_embeddings
        embeddings = torch.cat(embeds, dim=0)

        return embeddings


class PrompLangCLIPTextTransformer(CLIPTextTransformer):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.embeddings = PromptLangCLIPTextEmbeddings(config)
        self.transformers_version = version.parse(import_version('transformers'))

    def process_attention_mask(self, hidden_states, attention_mask, bsz, seq_len):
        # Parse the transformer version
        input_shape = torch.Size([bsz, seq_len])

        v4_30 = version.parse('4.30.0')
        v4_35 = version.parse('4.35')
        if self.transformers_version < v4_30:
            print("Using transformers < 4.30.0")
            causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
                hidden_states.device)
        elif v4_30 <= self.transformers_version < v4_35:
            print("Using transformers >= 4.30.0 and <= 4.34.*")
            from transformers.models.clip.modeling_clip import _make_causal_mask
            causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        else:
            print("Using transformers >= 4.35")
            from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
            causal_attention_mask = _create_4d_causal_attention_mask(
                input_shape, hidden_states.dtype, device=hidden_states.device
            )

        # Expand attention_mask if it exists
        if attention_mask is not None:
            # Import _expand_mask or _prepare_4d_attention_mask based on version
            if self.transformers_version < v4_35:
                from transformers.models.clip.modeling_clip import _expand_mask
                attention_mask = _expand_mask(attention_mask, hidden_states.dtype)
            else:
                from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
                attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        return causal_attention_mask, attention_mask

    def forward(
            self,
            input_ids: Optional[List[List[SegOrAction]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        # input_shape = input_ids.size()
        # input_ids = input_ids.view(-1, input_shape[-1])

        for batch_idx, batch in enumerate(input_ids):
            for seg_or_action in batch:
                if isinstance(seg_or_action, Action):
                    seg_or_action.process_with_transformer(
                        self, self.embeddings.token_embedding
                    )

        hidden_states = self.embeddings(input_dicts=input_ids)

        bsz = len(input_ids)
        seq_len = hidden_states.shape[1]

        causal_attention_mask, attention_mask = self.process_attention_mask(hidden_states, attention_mask, bsz, seq_len)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)


        # Hacky way to get idx of first EOT token
        eot_idx = [1]
        for batch in input_ids[1:]:
            idx = 0
            for seg_or_action in batch:
                if isinstance(seg_or_action, Action):
                    idx += seg_or_action.token_length()
                else:
                    if seg_or_action.text == "__PAD__" or seg_or_action.text == "[EOT]":
                        break

                    # Is a segment, and isn't the pad segment
                    idx += seg_or_action.token_length()
            eot_idx.append(idx)
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        # TODO: Get the index of the first EOT token
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            eot_idx
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# This is necessary to pass the PromptLangCLIPTextTransformer
class PromptLangTextModel(CLIPTextModel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = PrompLangCLIPTextTransformer(config)

    def forward(
            self,
            input_ids: Optional[List[List[SegOrAction]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
