import torch
from torch import Tensor
from torch.nn import Embedding

from .action_utils import concat_embeddings, get_total_length
from .base import SingleArgAction


class NormAction(SingleArgAction):
    grammar = 'norm(" arg+ ")"'

    display_name = "Normalize"
    action_name = "norm"
    description = "Normalizes the provided segments or actions."
    usage_examples = [
        "norm(cat)",
        "sum(cat|norm(sum(tiger|fish)))",
    ]

    def token_length(self) -> int:
        return get_total_length(self.arg)

    def get_result(self, embedding_module: Embedding) -> Tensor:
        embeddings = concat_embeddings(self.arg, embedding_module)
        return torch.div(embeddings, torch.norm(embeddings, dim=-1, keepdim=True))
