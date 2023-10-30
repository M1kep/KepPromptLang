from typing import Tuple, List

import torch
from torch.nn import Embedding

from custom_nodes.KepPromptLang.lib.action.base import (
    Action,
    PostModifiers,
    MultiArgAction,
)
from custom_nodes.KepPromptLang.lib.actions.types import SegOrAction


class PosScaleAction(MultiArgAction):
    grammar = 'posScale(" arg+ ")"'
    name = "posScale"
    chars = ["[", "]"]

    def __init__(self, args: List[List[SegOrAction]]) -> None:
        super().__init__(args)
        if len(args) != 2:
            raise ValueError("PosScale action should have exactly two arguments")

        self.target_arg = args[0]
        self._parse_multiplier(args[1])

    def _parse_multiplier(self, arg: List[SegOrAction]) -> None:
        if len(arg) != 1:
            raise ValueError(
                "PosScale actions multiplier should have exactly one segment"
            )

        multiplier_seg_or_action = arg[0]

        if isinstance(multiplier_seg_or_action, Action):
            raise ValueError("PosScale actions multiplier must be a number")

        try:
            self.parsed_multiplier = float(multiplier_seg_or_action.text)
        except ValueError:
            raise ValueError(
                "PosScale action should have an integer/float as the multiplier"
            )

    def token_length(self) -> int:
        """
        PosScale modifies the posional embeddings of the base segment, so the length is the length of the base segment
        :return:
        """
        total_length = 0
        for seg_or_action in self.target_arg:
            total_length += seg_or_action.token_length()

        return total_length

    def get_result(self, embedding_module: Embedding) -> Tuple[torch.Tensor, PostModifiers]:
        all_embeddings = []
        for seg_or_action in self.target_arg:
            if isinstance(seg_or_action, Action):
                all_embeddings.append(seg_or_action.get_result(embedding_module))
            else:
                all_embeddings.append(seg_or_action.get_embeddings(embedding_module))

        target_embeddings = torch.cat(all_embeddings, dim=1)
        return target_embeddings, {"position_embed_scale": self.parsed_multiplier}
