from typing import List, Union

import torch
from torch.nn import Embedding

from custom_nodes.KepPromptLang.lib.action.base import MultiArgAction, Action
from custom_nodes.KepPromptLang.lib.actions.action_utils import get_embedding
from custom_nodes.KepPromptLang.lib.actions.types import SegOrAction
from custom_nodes.KepPromptLang.lib.actions.utils import slerp
from custom_nodes.KepPromptLang.lib.parser.prompt_segment import PromptSegment


class AverageAction(MultiArgAction):
    grammar = 'avg(" arg "|" arg "|" arg ")"'
    chars = ["+", "+"]

    display_name = "Average"
    action_name = "avg"
    description = "Performs a weighted average between two segments or actions. The recommended weight is 0 - 1."
    usage_examples = [
        "avg(The cat is|The dog is|0.5)",
        "avg(Cat|Dog|0.5)",
    ]


    def __init__(self, args: List[List[Union[PromptSegment, Action]]]) -> None:
        super().__init__(args)

        if len(args) != 3:
            raise ValueError("Average action should have exactly three arguments(2 vectors and a weight)")

        self.first_arg = args[0]
        self.second_arg = args[1]
        self._parse_weight(args[2])

        self._validate_args()


    def _parse_weight(self, arg: List[SegOrAction]) -> None:
        if len(arg) != 1:
            raise ValueError("Average weight should have exactly one segment")

        weight_seg_or_action = arg[0]

        if isinstance(weight_seg_or_action, Action):
            raise ValueError("Average weight should not have an action as an argument")

        try:
            self.parsed_weight = float(weight_seg_or_action.text)
        except ValueError:
            raise ValueError("Average should have an integer/float as the weight")

    def _validate_args(self) -> None:
        first_arg_token_length = sum(seg_or_action.token_length() for seg_or_action in self.first_arg)
        second_arg_token_length = sum(seg_or_action.token_length() for seg_or_action in self.second_arg)
        if first_arg_token_length != second_arg_token_length:
            raise ValueError(f"Average start and end arguments should have the same length. Got {start_arg_token_length} and {end_arg_token_length}")

        if self.parsed_weight < 0 or self.parsed_weight > 1:
            print(f"WARNING: Average weight should be between 0 and 1. Got {self.parsed_weight}")

    def token_length(self) -> int:
        # Average interpolates between the embeddings of the start and end segments, so the length is the length of the start segment
        return sum(seg_or_action.token_length() for seg_or_action in self.first_arg)

    def get_result(self, embedding_module: Embedding) -> torch.Tensor:
        # Calculate the embeddings for the start segment
        all_start_embeddings = [
            get_embedding(seg_or_action, embedding_module)
            for seg_or_action in self.first_arg
        ]

        start_embedding = torch.cat(all_start_embeddings, dim=1)

        # Calculate the embeddings for the end segment
        all_end_embeddings = [
            get_embedding(seg_or_action, embedding_module)
            for seg_or_action in self.second_arg
        ]

        end_embedding = torch.cat(all_end_embeddings, dim=1)

        # Perform the weighted average
        result = start_embedding * (1 - self.parsed_weight) + end_embedding * self.parsed_weight

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
