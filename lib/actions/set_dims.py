from typing import List, Tuple

from torch import Tensor
from torch.nn import Embedding

from .action_utils import concat_embeddings, get_total_length
from .base import MultiArgAction
from .scale_dims import _parse_dim_value_pairs
from .types import SegOrAction


class SetDims(MultiArgAction):
    grammar = 'setDims(" arg ("|" arg)* ")"'

    display_name = "Set Dimensions"
    action_name = "setDims"
    description = "Sets the specified dimensions of the input embeddings to the specified value"
    usage_examples = [
        "The setDims(cat|4, -0.01253|76, 1.2) is happy",
    ]

    def __init__(self, args: List[List[SegOrAction]]):
        super().__init__(args)
        self.base_arg = args[0]
        self.value_args: List[Tuple[int, float]] = _parse_dim_value_pairs(args[1:], action_name="SetDims")

    def token_length(self) -> int:
        return get_total_length(self.base_arg)

    def get_result(self, embedding_module: Embedding) -> Tensor:
        embeddings = concat_embeddings(self.base_arg, embedding_module)
        for dim, value in self.value_args:
            embeddings[0, :, dim] = value
        return embeddings
