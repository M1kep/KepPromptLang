from typing import List

import torch
from torch import Tensor
from torch.nn import Embedding

from .action_utils import parse_numeric_arg
from .base import MultiArgAction
from .types import SegOrAction


class RandAction(MultiArgAction):
    grammar = 'rand(" arg ")"'

    display_name = "Random Embedding"
    action_name = "rand"
    description = (
        "Returns a random embedding of the specified token length, "
        "with the values optionally bounded by the second and third arguments."
    )
    usage_examples = [
        "A rand(1) cat",
        "A rand(1|-1|1) cat",
    ]

    def __init__(self, args: List[List[SegOrAction]]) -> None:
        super().__init__(args)
        if len(args) not in (1, 3):
            raise ValueError("Random action expects exactly one or three arguments")

        self.parsed_token_length = parse_numeric_arg(
            args[0], action_name="Random", role="first argument (token length)", cast=int
        )
        if len(args) == 3:
            self.range_min = parse_numeric_arg(
                args[1], action_name="Random", role="second argument (min)", cast=int
            )
            self.range_max = parse_numeric_arg(
                args[2], action_name="Random", role="third argument (max)", cast=int
            )
            if self.range_min > self.range_max:
                raise ValueError("Random action min must be <= max")
        else:
            self.range_min = 0
            self.range_max = 1

    def token_length(self) -> int:
        return self.parsed_token_length

    def get_result(self, embedding_module: Embedding) -> Tensor:
        return torch.empty(
            1, self.parsed_token_length, embedding_module.embedding_dim
        ).uniform_(self.range_min, self.range_max)
