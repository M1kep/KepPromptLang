from typing import List

from torch import Tensor
from torch.nn import Embedding

from .action_utils import concat_embeddings, get_total_length, parse_numeric_arg
from .base import MultiArgAction
from .types import SegOrAction


class AverageAction(MultiArgAction):
    grammar = 'avg(" arg "|" arg "|" arg ")"'

    display_name = "Average"
    action_name = "avg"
    description = "Performs a weighted average between two segments or actions. The recommended weight is 0 - 1."
    usage_examples = [
        "avg(The cat is|The dog is|0.5)",
        "avg(Cat|Dog|0.5)",
    ]

    def __init__(self, args: List[List[SegOrAction]]) -> None:
        super().__init__(args)
        if len(args) != 3:
            raise ValueError("Average action expects exactly three arguments (2 vectors and a weight)")

        self.first_arg = args[0]
        self.second_arg = args[1]
        self.parsed_weight = parse_numeric_arg(
            args[2], action_name="Average", role="weight", cast=float
        )

        first_len = get_total_length(self.first_arg)
        second_len = get_total_length(self.second_arg)
        if first_len != second_len:
            raise ValueError(
                f"Average start and end arguments should have the same length. Got {first_len} and {second_len}"
            )

        if self.parsed_weight < 0 or self.parsed_weight > 1:
            print(f"WARNING: Average weight should be between 0 and 1. Got {self.parsed_weight}")

    def token_length(self) -> int:
        return get_total_length(self.first_arg)

    def get_result(self, embedding_module: Embedding) -> Tensor:
        start = concat_embeddings(self.first_arg, embedding_module)
        end = concat_embeddings(self.second_arg, embedding_module)
        return start * (1 - self.parsed_weight) + end * self.parsed_weight
