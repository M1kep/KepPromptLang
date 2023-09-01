from typing import List

import torch
from torch.nn import Embedding

from custom_nodes.KepPromptLang.lib.action.base import (
    Action,
    MultiArgAction,
)
from custom_nodes.KepPromptLang.lib.actions.types import SegOrAction
from custom_nodes.KepPromptLang.lib.parser.registration import register_action


class MultiplyAction(MultiArgAction):
    grammar = 'mult(" arg+ ")"'
    name = "mult"
    chars = ["[", "]"]

    def __init__(self, args: List[List[SegOrAction]]) -> None:
        super().__init__(args)
        if len(args) != 2:
            raise ValueError("Multiply action should have exactly two arguments")

        self.target_arg = args[0]
        self._parse_multiplier(args[1])

    def _parse_multiplier(self, arg: List[SegOrAction]) -> None:
        if len(arg) != 1:
            raise ValueError("Multiply actions multiplier should have exactly one segment")

        multiplier_seg_or_action = arg[0]

        if isinstance(multiplier_seg_or_action, Action):
            raise ValueError("Multiply actions multiplier must be a number")

        try:
            self.parsed_multiplier = float(multiplier_seg_or_action.text)
        except ValueError:
            raise ValueError("Multiply action should have an integer/float as the multiplier")

    def token_length(self) -> int:
        """
        Mult multiplies the embeddings of the base segment, so the length is the length of the base segment
        :return:
        """
        total_length = 0
        for seg_or_action in self.target_arg:
            total_length += seg_or_action.token_length()

        return total_length

    def get_result(self, embedding_module: Embedding) -> torch.Tensor:
        all_embeddings = []
        for seg_or_action in self.target_arg:
            if isinstance(seg_or_action, Action):
                all_embeddings.append(seg_or_action.get_result(embedding_module))
            else:
                all_embeddings.append(seg_or_action.get_embeddings(embedding_module))

        target_embeddings = torch.cat(all_embeddings, dim=1)
        return target_embeddings * self.parsed_multiplier

