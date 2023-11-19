from typing import List, Union

import torch
from torch.nn import Embedding

from custom_nodes.KepPromptLang.lib.action.base import MultiArgAction, Action
from custom_nodes.KepPromptLang.lib.actions.action_utils import get_embedding
from custom_nodes.KepPromptLang.lib.parser.prompt_segment import PromptSegment


class ScaleDims(MultiArgAction):
    grammar = 'scaleDims(" arg ("|" arg)* ")"'
    chars = ["-", "-"]

    display_name = "Scale Dimensions"
    action_name = "scaleDims"
    description = "Scales the specified dimensions of the input embeddings by the specified amount"
    usage_examples = [
        "The scaleDims(cat|4,1.5|76,1.2) is happy",
    ]

    def __init__(self, args: List[List[Union[PromptSegment, Action]]]):
        super().__init__(args)

        self.base_arg = args[0]
        self._parse_scale_args(args[1:])


    def _parse_scale_args(self, args: List[List[Union[PromptSegment, Action]]]) -> None:
        # scaleDims scale args should have format of "<dim>,<scale>" where dim is the dimension to scale and scale is the amount to scale it by
        # scaleDims(some words|4,1.5|76,1.2)
        self.scale_args = []
        for arg in args:
            if isinstance(arg, Action):
                raise ValueError("ScaleDims scale args must be in the format of <dim>,<scale>(e.g. 4,1.5) but got an action")

            if len(arg) != 1:
                raise ValueError("ScaleDims scale args must be in the format of <dim>,<scale>(e.g. 4,1.5) but got multiple segments")
            extracted_arg = arg[0]
            assert isinstance(extracted_arg, PromptSegment)

            if "," not in extracted_arg.text:
                raise ValueError("ScaleDims scale args must be in the format of <dim>,<scale>(e.g. 4,1.5) but got a segment with no comma: " + extracted_arg.text)

            # Split prompt segment into text and scale args
            dim, scale = extracted_arg.text.split(",")
            try:
                # TODO: Check that dim is within the bounds of the embedding
                parsed_dim = int(dim)
            except ValueError:
                raise ValueError("ScaleDims scale args must be in the format of <dim>,<scale>(e.g. 4,1.5) but got a segment with a non-integer dim: " + str(dim))

            try:
                parsed_scale = float(scale)
            except ValueError:
                raise ValueError("ScaleDims scale args must be in the format of <dim>,<scale>(e.g. 4,1.5) but got a segment with a non-float scale: " + str(scale))

            self.scale_args.append((parsed_dim, parsed_scale))


    def token_length(self) -> int:
        # scaleDims modifies the embeddings of the base segment, so the length is the length of the base segment
        return sum(seg_or_action.token_length() for seg_or_action in self.base_arg)

    def get_result(self, embedding_module: Embedding) -> torch.Tensor:
        # Calculate the embeddings for the base segment
        all_base_embeddings = [
            get_embedding(seg_or_action, embedding_module)
            for seg_or_action in self.base_arg
        ]

        base_embeddings = torch.cat(all_base_embeddings, dim=1)
        for dim, scale in self.scale_args:
            base_embeddings[0, :, dim] *= scale

        return base_embeddings
