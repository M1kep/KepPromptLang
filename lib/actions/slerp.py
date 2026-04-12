from typing import List

from torch import Tensor
from torch.nn import Embedding

from .action_utils import concat_embeddings, get_total_length, parse_numeric_arg
from .base import MultiArgAction
from .types import SegOrAction
from .utils import slerp


class SlerpAction(MultiArgAction):
    grammar = 'slerp(" arg "|" arg "|" arg ")"'

    display_name = "Slerp"
    action_name = "slerp"
    description = (
        "Performs a slerp (interpolation) between two segments or actions, with the given weight. "
        "The recommended weight is 0 - 1."
    )
    usage_examples = [
        "The slerp(cat|dog|0.5) is happy",
    ]

    def __init__(self, args: List[List[SegOrAction]]) -> None:
        super().__init__(args)
        if len(args) != 3:
            raise ValueError("Slerp action expects exactly three arguments (2 vectors and a weight)")

        self.start_argument = args[0]
        self.end_argument = args[1]
        self.parsed_weight = parse_numeric_arg(
            args[2], action_name="Slerp", role="weight", cast=float
        )

        start_len = get_total_length(self.start_argument)
        end_len = get_total_length(self.end_argument)
        if start_len != end_len:
            raise ValueError(
                f"Slerp start and end arguments should have the same length. Got {start_len} and {end_len}"
            )

        if self.parsed_weight < 0 or self.parsed_weight > 1:
            print(f"WARNING: Slerp weight should be between 0 and 1. Got {self.parsed_weight}")

    def token_length(self) -> int:
        return get_total_length(self.start_argument)

    def get_result(self, embedding_module: Embedding) -> Tensor:
        start = concat_embeddings(self.start_argument, embedding_module)
        end = concat_embeddings(self.end_argument, embedding_module)
        return slerp(self.parsed_weight, start, end)
