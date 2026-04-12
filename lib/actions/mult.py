from typing import List

from torch import Tensor
from torch.nn import Embedding

from .action_utils import concat_embeddings, get_total_length, parse_numeric_arg
from .base import MultiArgAction
from .types import SegOrAction


class MultiplyAction(MultiArgAction):
    grammar = 'mult(" arg+ ")"'

    display_name = "Multiply"
    action_name = "mult"
    description = "Multiplies the provided segments or actions by the multiplier."
    usage_examples = [
        "mult(The cat is|2.5)",
        "mult(Cat|-1)",
    ]

    def __init__(self, args: List[List[SegOrAction]]) -> None:
        super().__init__(args)
        if len(args) != 2:
            raise ValueError("Multiply action expects exactly two arguments")

        self.target_arg = args[0]
        self.parsed_multiplier = parse_numeric_arg(
            args[1], action_name="Multiply", role="multiplier", cast=float
        )

    def token_length(self) -> int:
        return get_total_length(self.target_arg)

    def get_result(self, embedding_module: Embedding) -> Tensor:
        return concat_embeddings(self.target_arg, embedding_module) * self.parsed_multiplier
