from typing import List, Union

import torch
from torch.nn import Embedding

from custom_nodes.KepPromptLang.lib.action.base import MultiArgAction, Action
from custom_nodes.KepPromptLang.lib.actions.action_utils import get_embedding
from custom_nodes.KepPromptLang.lib.actions.types import SegOrAction
from custom_nodes.KepPromptLang.lib.actions.utils import slerp
from custom_nodes.KepPromptLang.lib.parser.prompt_segment import PromptSegment


class SlerpAction(MultiArgAction):
    grammar = 'slerp(" arg "|" arg "|" arg ")"'
    name = "slerp"
    chars = ["+", "+"]

    def __init__(self, args: List[List[Union[PromptSegment, Action]]]) -> None:
        super().__init__(args)

        if len(args) != 3:
            raise ValueError("Slerp action should have exactly three arguments(2 vectors and a weight)")

        self.start_argument = args[0]
        self.end_argument = args[1]
        self._parse_weight(args[2])

        self._validate_args()


    def _parse_weight(self, arg: List[SegOrAction]) -> None:
        if len(arg) != 1:
            raise ValueError("Slerp weight should have exactly one segment")

        weight_seg_or_action = arg[0]

        if isinstance(weight_seg_or_action, Action):
            raise ValueError("Slerp weight should not have an action as an argument")

        try:
            self.parsed_weight = float(weight_seg_or_action.text)
        except ValueError:
            raise ValueError("Slerp should have an integer/float as the weight")

    def _validate_args(self) -> None:
        start_arg_token_length = sum(seg_or_action.token_length() for seg_or_action in self.start_argument)
        end_arg_token_length = sum(seg_or_action.token_length() for seg_or_action in self.end_argument)
        if start_arg_token_length != end_arg_token_length:
            raise ValueError(f"Slerp start and end arguments should have the same length. Got {start_arg_token_length} and {end_arg_token_length}")

        if self.parsed_weight < 0 or self.parsed_weight > 1:
            print(f"WARNING: Slerp weight should be between 0 and 1. Got {self.parsed_weight}")

    def token_length(self) -> int:
        # Slerp interpolates between the embeddings of the start and end segments, so the length is the length of the start segment
        return sum(seg_or_action.token_length() for seg_or_action in self.start_argument)

    def get_result(self, embedding_module: Embedding) -> torch.Tensor:
        # Calculate the embeddings for the start segment
        all_start_embeddings = [
            get_embedding(seg_or_action, embedding_module)
            for seg_or_action in self.start_argument
        ]

        start_embedding = torch.cat(all_start_embeddings, dim=1)

        # Calculate the embeddings for the end segment
        all_end_embeddings = [
            get_embedding(seg_or_action, embedding_module)
            for seg_or_action in self.end_argument
        ]

        end_embedding = torch.cat(all_end_embeddings, dim=1)

        # Perform the slerp
        result = slerp(self.parsed_weight, start_embedding, end_embedding)

        return result

    # def __repr__(self):
    #     return f"sum(\n\tbase_segment={self.base_segment},\n\targs={self.args}\n)"
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
