from typing import List

import torch
from torch import Tensor
from torch.nn import Embedding

from .action_utils import concat_embeddings, get_total_length, parse_numeric_arg
from .base import MultiArgAction
from .types import SegOrAction


class NoiseAction(MultiArgAction):
    grammar = 'noise(" arg "|" arg ")"'

    display_name = "Noise"
    action_name = "noise"
    description = "Adds Gaussian noise (mean 0, given std) to the embeddings of the first argument."
    usage_examples = [
        "A noise(cat|0.05) on a sunny day",
    ]

    def __init__(self, args: List[List[SegOrAction]]) -> None:
        super().__init__(args)
        if len(args) != 2:
            raise ValueError("noise expects exactly two arguments: noise(a|std)")
        self.a_arg = args[0]
        self.std = parse_numeric_arg(args[1], action_name="noise", role="std", cast=float)

    def token_length(self) -> int:
        return get_total_length(self.a_arg)

    def get_result(self, embedding_module: Embedding) -> Tensor:
        a = concat_embeddings(self.a_arg, embedding_module)
        return a + torch.randn_like(a) * self.std
