from typing import List, Union

import torch
from torch.nn import Embedding

from custom_nodes.KepPromptLang.lib.action.base import MultiArgAction, Action
from custom_nodes.KepPromptLang.lib.actions.action_utils import (
    get_embedding,
    get_embedding_for_segments,
)
from custom_nodes.KepPromptLang.lib.actions.types import SegOrAction
from custom_nodes.KepPromptLang.lib.actions.utils import slerp
from custom_nodes.KepPromptLang.lib.parser.prompt_segment import PromptSegment


class ProjectAction(MultiArgAction):
    grammar = 'project(" arg "|" arg ("|" arg ")?)"'
    name = "project"
    chars = ["+", "+"]

    weight = 1.0

    def __init__(self, args: List[List[Union[PromptSegment, Action]]]) -> None:
        super().__init__(args)

        num_args = len(args)
        if num_args != 3 and num_args != 2:
            raise ValueError(
                "Project action should have exactly three arguments(2 vectors and a weight)"
            )

        self.source_argument = args[0]
        self.source_argument_token_length = sum(
            seg_or_action.token_length() for seg_or_action in self.source_argument
        )
        self.onto_argument = args[1]
        self.onto_argument_token_length = sum(
            seg_or_action.token_length() for seg_or_action in self.onto_argument
        )

        if num_args == 3:
            self._parse_weight(args[2])

        self._validate_args()

    def _parse_weight(self, arg: List[SegOrAction]) -> None:
        if len(arg) != 1:
            raise ValueError("Project weight should have exactly one segment")

        weight_seg_or_action = arg[0]

        if isinstance(weight_seg_or_action, Action):
            raise ValueError("Project weight should not have an action as an argument")

        try:
            self.weight = float(weight_seg_or_action.text)
        except ValueError:
            raise ValueError("Project should have an integer/float as the weight")

    def _validate_args(self) -> None:
        if (
            self.source_argument_token_length != self.onto_argument_token_length
            and self.onto_argument_token_length != 1
        ):
            raise ValueError(
                f"Project source and target arguments should have the same token lengths, or target should be one token. Got {self.source_argument_token_length} source tokens and {self.onto_argument_token_length} target tokens"
            )

    def token_length(self) -> int:
        # Project projects the source onto the target, so the length of the result is the length of source
        return self.source_argument_token_length

    def get_result(self, embedding_module: Embedding) -> torch.Tensor:
        # Calculate the embeddings for the start segment
        source_embedding = get_embedding_for_segments(
            self.source_argument, embedding_module
        )
        onto_embedding = get_embedding_for_segments(
            self.onto_argument, embedding_module
        )

        # Perform the projection
        return torch.mul(
            torch.mul(source_embedding, onto_embedding)
            / torch.mul(onto_embedding, onto_embedding),
            onto_embedding,
        )

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
