import contextlib
import os
from typing import List

import torch
from transformers import CLIPTextConfig, modeling_utils

from comfy import model_management
import comfy.ops
from custom_nodes.ClipStuff.lib.action.base import Action
from custom_nodes.ClipStuff.lib.actions.types import SegOrAction
from custom_nodes.ClipStuff.lib.fun_clip_stuff import PromptLangTextModel
from custom_nodes.ClipStuff.lib.parser.prompt_segment import PromptSegment


# Methods with no comment can be assumed to be the same as comfy.sd1_clip.SD1ClipModel
class PromptLangClipModel(torch.nn.Module):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]

    def __init__(self, version="openai/clip-vit-large-patch14", device="cpu", max_length=77,
                 freeze=True, layer="last", layer_idx=None, textmodel_json_config=None,
                 textmodel_path=None, dtype=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.num_layers = 12
        if textmodel_path is not None:
            # Our transformer
            self.transformer = PromptLangTextModel.from_pretrained(textmodel_path)
        else:
            if textmodel_json_config is None:
                # TODO: Maybe re-use clip config?
                # Config could come from cond_stage_model.transformer.config
                # Copied clip_config
                textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_config.json")
            config = CLIPTextConfig.from_json_file(textmodel_json_config)
            self.num_layers = config.num_hidden_layers
            with comfy.ops.use_comfy_ops(device, dtype):
                with modeling_utils.no_init_weights():
                    # Our transformer
                    self.transformer = PromptLangTextModel(config)

        if dtype is not None:
            self.transformer.to(dtype)
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.empty_tokens = [[49406] + [49407] * 76]
        self.text_projection = torch.nn.Parameter(torch.eye(self.transformer.get_input_embeddings().weight.shape[1]))
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))

        self.layer_norm_hidden_state = True
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) <= self.num_layers
            self.clip_layer(layer_idx)
        self.layer_default = (self.layer, self.layer_idx)

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def clip_layer(self, layer_idx):
        if abs(layer_idx) >= self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def reset_clip_layer(self):
        self.layer = self.layer_default[0]
        self.layer_idx = self.layer_default[1]

    # Completely changed to support Segments and actions
    def set_up_textual_embeddings(self, tokens: List[List[SegOrAction]], current_embeds):
        next_new_token = token_dict_size = current_embeds.weight.shape[0] - 1
        embedding_weights = []

        # For each batch
        for batch in tokens:
            for seg_or_action in batch:
                if isinstance(seg_or_action, Action):
                    segments = seg_or_action.get_all_segments()
                else:
                    segments = [seg_or_action]

                for segment in segments:
                    tokens_temp = []
                    segment_length = segment.token_length()
                    for tid_or_tensor in segment.tokens:
                        if isinstance(tid_or_tensor, int):
                            if tid_or_tensor == token_dict_size:  # Is EOS token
                                tid_or_tensor = -1 # Set to -1 so that it can be replaced with the EOS token later
                            tokens_temp += [tid_or_tensor]
                        else:
                            if tid_or_tensor.shape[0] == current_embeds.weight.shape[1]:
                                embedding_weights += [tid_or_tensor]
                                tokens_temp += [next_new_token]
                                next_new_token += 1
                            else:
                                print("WARNING: shape mismatch when trying to apply embedding, embedding will be ignored",
                                      tid_or_tensor.shape[0], current_embeds.weight.shape[1])
                    if len(tokens_temp) < segment_length:
                        # Pretty sure this is only needed if the embedding is not the same size as the CLIP embedding
                        print("WARNING: segment length mismatch, padding with EOS token")
                        tokens_temp.extend([self.empty_tokens[0][-1] * (segment_length - len(tokens_temp))])
                    segment.tokens = tokens_temp

        n = token_dict_size
        if len(embedding_weights) > 0:
            # Create new embedding, with size of current embedding + number of new embeddings
            new_embedding = torch.nn.Embedding(next_new_token + 1, current_embeds.weight.shape[1],
                                               device=current_embeds.weight.device, dtype=current_embeds.weight.dtype)
            # Copy current embedding weights to new embedding
            new_embedding.weight[:token_dict_size] = current_embeds.weight[:-1]
            # Add new embeddings
            for embed in embedding_weights:
                new_embedding.weight[n] = embed
                n += 1

            # Set re-add the EOS token
            new_embedding.weight[n] = current_embeds.weight[-1]  # EOS embedding
            self.transformer.set_input_embeddings(new_embedding)


        for batch in tokens:
            for seg_or_action in batch:
                if isinstance(seg_or_action, Action):
                    segments = seg_or_action.get_all_segments()
                else:
                    segments = [seg_or_action]

                for segment in segments:
                    for tokenIdx in range(len(segment.tokens)):
                        if segment.tokens[tokenIdx] == -1:
                            segment.tokens[tokenIdx] = n

    # Support our set_up_textual_embeddings which modifies the input embeddings
    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        self.set_up_textual_embeddings(tokens, backup_embeds)
        # tokens = torch.LongTensor(tokens).to(device)

        if backup_embeds.weight.dtype != torch.float32:
            precision_scope = torch.autocast
        else:
            precision_scope = contextlib.nullcontext

        with precision_scope(model_management.get_autocast_device(device)):
            outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer == "hidden")
            self.transformer.set_input_embeddings(backup_embeds)

            if self.layer == "last":
                z = outputs.last_hidden_state
            elif self.layer == "pooled":
                z = outputs.pooler_output[:, None, :]
            else:
                z = outputs.hidden_states[self.layer_idx]
                if self.layer_norm_hidden_state:
                    z = self.transformer.text_model.final_layer_norm(z)

            pooled_output = outputs.pooler_output
            if self.text_projection is not None:
                pooled_output = pooled_output.float().to(self.text_projection.device) @ self.text_projection.float()
        return z.float(), pooled_output.float()

    def encode(self, tokens):
        return self(tokens)

    def load_sd(self, sd):
        if "text_projection" in sd:
            self.text_projection[:] = sd.pop("text_projection")
        if "text_projection.weight" in sd:
            self.text_projection[:] = sd.pop("text_projection.weight").transpose(0, 1)
        return self.transformer.load_state_dict(sd, strict=False)

    # Changed from comfy.sd1_clip.ClipTokenWeightEncoder
    # Changed to use PromptSegments
    def encode_token_weights(self, prompt_segments: List[List[SegOrAction]]):
        to_encode = [[PromptSegment(text="_Empty Batch_", tokens=self.empty_tokens[0])]]
        for batch in prompt_segments:
            to_encode.append(batch)

        out, pooled = self.encode(to_encode)
        z_empty = out[0:1]
        if pooled.shape[0] > 1:
            first_pooled = pooled[1:2]
        else:
            first_pooled = pooled[0:1]

        output = []
        for k in range(1, out.shape[0]):
            z = out[k:k + 1]
            # for i in range(len(z)):
            #     for j in range(len(z[i])):
            #         weight = token_dicts[k - 1][j][0].weight
            #         z[i][j] = (z[i][j] - z_empty[0][j]) * weight + z_empty[0][j]
            output.append(z)

        if (len(output) == 0):
            return z_empty.cpu(), first_pooled.cpu()
        return torch.cat(output, dim=-2).cpu(), first_pooled.cpu()
