from typing import List, Union

import torch
from torch.nn import Embedding

from custom_nodes.KepPromptLang.lib.action.base import MultiArgAction, Action
from custom_nodes.KepPromptLang.lib.actions.action_utils import get_embedding
from custom_nodes.KepPromptLang.lib.parser.prompt_segment import PromptSegment


class SetDims(MultiArgAction):
    grammar = 'setDims(" arg ("|" arg)* ")"'
    chars = ["-", "-"]

    display_name = "Set Dimensions"
    action_name = "setDims"
    description = "Sets the specified dimensions of the input embeddings to the specified value"
    usage_examples = [
        "The setDims(cat|4, -0.01253|76, 1.2) is happy"
    ]

    def __init__(self, args: List[List[Union[PromptSegment, Action]]]):
        super().__init__(args)

        self.base_arg = args[0]
        self._parse_value_args(args[1:])


    def _parse_value_args(self, args: List[List[Union[PromptSegment, Action]]]) -> None:
        # setDims args should have format of "<dim>,<value>" where dim is the dimension to set and value is the value to set it to
        # setDims(some words|4,-0.01254|76,1.2)
        self.value_args = []
        for arg in args:
            if isinstance(arg, Action):
                raise ValueError("SetDims value args must be in the format of <dim>,<value>(e.g. 4,1.5) but got an action")

            if len(arg) != 1:
                raise ValueError("SetDims value args must be in the format of <dim>,<value>(e.g. 4,1.5) but got multiple segments")
            extracted_arg = arg[0]
            assert isinstance(extracted_arg, PromptSegment)

            if "," not in extracted_arg.text:
                raise ValueError("SetDims value args must be in the format of <dim>,<value>(e.g. 4,1.5) but got a segment with no comma: " + extracted_arg.text)

            # Split prompt segment into text and value args
            dim, value = extracted_arg.text.split(",")
            try:
                # TODO: Check that dim is within the bounds of the embedding
                parsed_dim = int(dim)
            except ValueError:
                raise ValueError("SetDims value args must be in the format of <dim>,<value>(e.g. 4,1.5) but got a segment with a non-integer dim: " + str(dim))

            try:
                parsed_value = float(value)
            except ValueError:
                raise ValueError("SetDims value args must be in the format of <dim>,<value>(e.g. 4,1.5) but got a segment with a non-float scale: " + str(value))

            self.value_args.append((parsed_dim, parsed_value))


    def token_length(self) -> int:
        # setDims modifies the embeddings of the base segment, so the length is the length of the base segment
        return sum(seg_or_action.token_length() for seg_or_action in self.base_arg)

    def get_result(self, embedding_module: Embedding) -> torch.Tensor:
        # Calculate the embeddings for the base segment
        all_base_embeddings = [
            get_embedding(seg_or_action, embedding_module)
            for seg_or_action in self.base_arg
        ]

        base_embeddings = torch.cat(all_base_embeddings, dim=1)
        for dim, value in self.value_args:
            base_embeddings[0, :, dim] = value

        return base_embeddings
