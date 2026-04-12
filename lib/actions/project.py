from typing import List

import torch
from torch import Tensor
from torch.nn import Embedding

from .action_utils import concat_embeddings, get_total_length
from .base import MultiArgAction
from .types import SegOrAction


def _direction(args: List[SegOrAction], embedding_module: Embedding) -> Tensor:
    """Mean unit direction of an arg's embeddings: [1, 1, hidden]."""
    emb = concat_embeddings(args, embedding_module)
    mean = emb.mean(dim=1, keepdim=True)
    return torch.nn.functional.normalize(mean, dim=-1)


def _project(a: Tensor, b_hat: Tensor) -> Tensor:
    coeff = (a * b_hat).sum(dim=-1, keepdim=True)
    return coeff * b_hat


class ProjectAction(MultiArgAction):
    grammar = 'proj(" arg "|" arg ")"'

    display_name = "Project"
    action_name = "proj"
    description = "Projects the first argument onto the direction of the second (mean, unit-normalized)."
    usage_examples = [
        "proj(king|gender)",
        "diff(style|proj(style|photorealistic))",
    ]

    def __init__(self, args: List[List[SegOrAction]]) -> None:
        super().__init__(args)
        if len(args) != 2:
            raise ValueError("proj expects exactly two arguments: proj(a|b)")
        self.a_arg = args[0]
        self.b_arg = args[1]

    def token_length(self) -> int:
        return get_total_length(self.a_arg)

    def get_result(self, embedding_module: Embedding) -> Tensor:
        a = concat_embeddings(self.a_arg, embedding_module)
        return _project(a, _direction(self.b_arg, embedding_module))


class RejectAction(MultiArgAction):
    grammar = 'reject(" arg "|" arg ")"'

    display_name = "Reject"
    action_name = "reject"
    description = "Removes the component of the first argument along the direction of the second (a - proj(a|b))."
    usage_examples = [
        "reject(anime girl|anime)",
    ]

    def __init__(self, args: List[List[SegOrAction]]) -> None:
        super().__init__(args)
        if len(args) != 2:
            raise ValueError("reject expects exactly two arguments: reject(a|b)")
        self.a_arg = args[0]
        self.b_arg = args[1]

    def token_length(self) -> int:
        return get_total_length(self.a_arg)

    def get_result(self, embedding_module: Embedding) -> Tensor:
        a = concat_embeddings(self.a_arg, embedding_module)
        return a - _project(a, _direction(self.b_arg, embedding_module))
