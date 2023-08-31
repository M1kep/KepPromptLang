from typing import Union, List

import torch
from torch.nn import Embedding

from custom_nodes.KepPromptLang.lib.action.base import Action, MultiArgAction
from custom_nodes.KepPromptLang.lib.actions.action_utils import get_embedding
from custom_nodes.KepPromptLang.lib.parser.prompt_segment import PromptSegment
from custom_nodes.KepPromptLang.lib.parser.registration import register_action


class DiffAction(MultiArgAction):
    grammar = 'diff(" arg ("|" arg)* ")"'
    name = "diff"
    chars = ["-", "-"]

    def __init__(self, args: List[List[Union[PromptSegment, Action]]]):
        super().__init__(args)
        self.base_arg = args[0]
        self.additional_args = args[1:]


    def token_length(self) -> int:
        # Sum adds to the embeddings of the base segment, so the length is the length of the base segment
        return sum(seg_or_action.token_length() for seg_or_action in self.base_arg)

    def get_result(self, embedding_module: Embedding) -> torch.Tensor:
        # Calculate the embeddings for the base segment
        all_base_embeddings = [
            get_embedding(seg_or_action, embedding_module)
            for seg_or_action in self.base_arg
        ]

        result = torch.cat(all_base_embeddings, dim=1)

        for arg in self.additional_args:
            all_arg_embeddings = [
                get_embedding(seg_or_action, embedding_module) for seg_or_action in arg
            ]

            arg_embedding = torch.cat(all_arg_embeddings, dim=1)

            if (
                arg_embedding.shape[-2] == 1
                or result.shape[-2] == arg_embedding.shape[-2]
            ):
                result = result.sub(arg_embedding)
            else:
                print(
                    "WARNING: shape mismatch when trying to apply sum, arg will be averaged"
                )
                result = result.sub(torch.mean(arg_embedding, dim=1, keepdim=True))

        return result

    # def __repr__(self):
    #     return f"sum(\n\tbase_segment={self.base_segment},\n\tadditional_args={self.additional_args}\n)"
    def __repr__(self) -> str:
        return f"sum({', '.join(map(str, self.additional_args))})"

    def depth_repr(self, depth=1):
        out = "NudgeAction(\n"
        if isinstance(self.base_arg, Action):
            base_segment_repr = self.base_arg.depth_repr(depth + 1)
            out += "\t" * depth + f"base_segment={base_segment_repr}\n"
        else:
            out += "\t" * depth + f"base_segment={self.base_arg.depth_repr()},\n"

        if isinstance(self.additional_args, Action):
            target_repr = self.additional_args.depth_repr(depth + 1)
            out += "\t" * depth + f"target={target_repr},\n"
        else:
            out += "\t" * depth + f"target={self.additional_args.depth_repr()},\n"
        out += "\t" * depth + f"weight={self.weight},\n"
        out += "\t" * (depth - 1) + ")"
        return out

