from typing import List

from torch import Tensor
from torch.nn import Embedding

from .action_utils import add_with_broadcast, concat_embeddings
from .base import MultiArgAction
from .types import SegOrAction


class DiffAction(MultiArgAction):
    grammar = 'diff(" arg ("|" arg)* ")"'

    display_name = "Difference"
    action_name = "diff"
    description = (
        "Subtracts the segments in the order they are given. "
        "The first segment is subtracted from the second, then the third from the result, and so on."
    )
    usage_examples = [
        "diff(The cat is|The dog is)",
        "diff(Cat|Dog)",
        "sum(diff(king|man)|woman)",
    ]

    def __init__(self, args: List[List[SegOrAction]]) -> None:
        super().__init__(args)
        self.base_arg = args[0]
        self.additional_args = args[1:]

    def token_length(self) -> int:
        return sum(s.token_length() for s in self.base_arg)

    def get_result(self, embedding_module: Embedding) -> Tensor:
        result = concat_embeddings(self.base_arg, embedding_module)
        for arg in self.additional_args:
            arg_embedding = concat_embeddings(arg, embedding_module)
            result = add_with_broadcast(result, arg_embedding, op="sub")
        return result
