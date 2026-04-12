from typing import List

import torch
from torch import Tensor
from torch.nn import Embedding

from .action_utils import concat_embeddings, parse_numeric_arg
from .base import MultiArgAction
from .types import SegOrAction


class NearestAction(MultiArgAction):
    grammar = 'nearest(" arg ("|" arg)? ")"'

    display_name = "Nearest Vocab"
    action_name = "nearest"
    description = (
        "Snaps a computed vector to the k nearest real vocabulary tokens (by cosine similarity), "
        "returning their embeddings concatenated. The input is mean-pooled before lookup."
    )
    usage_examples = [
        "nearest(sum(diff(king|man)|woman))",
        "nearest(sum(red|blue)|3)",
    ]

    def __init__(self, args: List[List[SegOrAction]]) -> None:
        super().__init__(args)
        if len(args) not in (1, 2):
            raise ValueError("nearest expects one or two arguments: nearest(expr) or nearest(expr|k)")
        self.expr_arg = args[0]
        self.k = parse_numeric_arg(args[1], action_name="nearest", role="k", cast=int) if len(args) == 2 else 1

    def token_length(self) -> int:
        return self.k

    def get_result(self, embedding_module: Embedding) -> Tensor:
        weight = embedding_module.weight.to(torch.float32)
        weight_norm = torch.nn.functional.normalize(weight, dim=-1)

        expr = concat_embeddings(self.expr_arg, embedding_module).to(torch.float32)
        query = torch.nn.functional.normalize(expr.mean(dim=1), dim=-1)

        sims = query @ weight_norm.T
        top_ids = sims.topk(self.k, dim=-1).indices.squeeze(0)
        return weight[top_ids].unsqueeze(0)
