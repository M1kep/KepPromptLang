from typing import List

import torch
from torch import Tensor
from torch.nn import Embedding

from .action_utils import concat_embeddings, get_total_length
from .base import MultiArgAction
from .types import SegOrAction


class RenormAction(MultiArgAction):
    grammar = 'renorm(" arg "|" arg ")"'

    display_name = "Renormalize"
    action_name = "renorm"
    description = "Rescales the first argument so each token's L2 norm matches the (mean) L2 norm of the reference."
    usage_examples = [
        "renorm(sum(king|neg(man)|woman)|queen)",
    ]

    def __init__(self, args: List[List[SegOrAction]]) -> None:
        super().__init__(args)
        if len(args) != 2:
            raise ValueError("renorm expects exactly two arguments: renorm(a|ref)")
        self.a_arg = args[0]
        self.ref_arg = args[1]

    def token_length(self) -> int:
        return get_total_length(self.a_arg)

    def get_result(self, embedding_module: Embedding) -> Tensor:
        a = concat_embeddings(self.a_arg, embedding_module)
        ref = concat_embeddings(self.ref_arg, embedding_module)
        a_norm = torch.norm(a, dim=-1, keepdim=True).clamp(min=1e-8)
        ref_norm = torch.norm(ref, dim=-1, keepdim=True).mean()
        return a * (ref_norm / a_norm)
