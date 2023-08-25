from typing import Optional, Tuple, Union

import torch
from transformers import CLIPTextConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import _expand_mask, CLIPTextEmbeddings, CLIPTextTransformer, \
    CLIPTextModel

from custom_nodes.ClipStuff.lib.action.base import Action
from custom_nodes.ClipStuff.lib.actions.types import SegOrAction

def slerp(val, low, high):
    low = low.unsqueeze(0)
    high = high.unsqueeze(0)
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

class PromptLangCLIPTextEmbeddings(CLIPTextEmbeddings):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)

    def forward(
            self,
            input_dicts: Optional[list[list[SegOrAction]]] = None,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:

        if input_dicts is None:
            raise ValueError("You have to specify input_dicts")

        batches = []
        for batch_idx, batch in enumerate(input_dicts):
            results = []
            for seg_or_action in batch:
                if isinstance(seg_or_action, Action):
                    results.append(seg_or_action.get_result(self.token_embedding))
                else:
                    results.append(seg_or_action.get_embeddings(self.token_embedding))
            batches.append(results)

        seq_length = batches[0][0].shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # if inputs_embeds is None:
        #     inputs_embeds = self.token_embedding(input_ids)

        # for batch_idx, batch in enumerate(input_dicts):
        #     for token_idx, token in enumerate(batch):
        #         if token[0].nudge_id is not None:
        #             nudged_embed = inputs_embeds[batch_idx, token_idx][:] + self.token_embedding(torch.LongTensor([token[0].nudge_id]).to(torch.device('cpu')))[0]
        #             if token[0].nudge_index_start is not None and token[0].nudge_index_stop is not None:
        #                 nudge_start = token[0].nudge_index_start
        #                 nudge_end = token[0].nudge_index_stop
        #             else:
        #                 nudge_start = 0
        #                 nudge_end = 768
        #             inputs_embeds[batch_idx, token_idx][nudge_start:nudge_end] = (slerp(token[0].nudge_weight, inputs_embeds[batch_idx, token_idx][:], nudged_embed)[0][nudge_start:nudge_end])
        #         elif token[0].arith_ops is not None:
        #             for op, id_list in token[0].arith_ops.items():
        #                 if op == '+':
        #                     for this_id in id_list:
        #                         inputs_embeds[batch_idx, token_idx] += self.token_embedding(torch.LongTensor([this_id]).to(torch.device('cpu')))[0]
        #                 elif op == '-':
        #                     for this_id in id_list:
        #                         inputs_embeds[batch_idx, token_idx] -= self.token_embedding(torch.LongTensor([this_id]).to(torch.device('cpu')))[0]

        embeds = []
        for batch in batches:
            if len(batch) == 1:
                embeds.append(batch[0])
            else:
                embeds.append(torch.cat(batch, dim=-2))

        position_embeddings = self.position_embedding(position_ids)
        embeddings = torch.cat(embeds, dim=0) + position_embeddings

        return embeddings


class PrompLangCLIPTextTransformer(CLIPTextTransformer):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.embeddings = PromptLangCLIPTextEmbeddings(config)

    def forward(
            self,
            input_ids: Optional[list[list[SegOrAction]]] = None,
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

        hidden_states = self.embeddings(input_dicts=input_ids)

        bsz = len(input_ids)
        # TODO: Properly gather this
        seq_len = 77
        # bsz, seq_len = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

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
                    if seg_or_action.text == '__PAD__':
                        break
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
            input_ids: Optional[list[list[SegOrAction]]] = None,
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
