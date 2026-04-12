from torch import Tensor
from torch.nn import Embedding

from .action_utils import concat_embeddings, get_total_length
from .base import SingleArgAction


class NegAction(SingleArgAction):
    grammar = 'neg(" arg+ ")"'

    display_name = "Negate"
    action_name = "neg"
    description = "Negates the provided segments or actions."
    usage_examples = [
        "neg(cat)",
        "sum(king|neg(man)|women)",
    ]

    def token_length(self) -> int:
        return get_total_length(self.arg)

    def get_result(self, embedding_module: Embedding) -> Tensor:
        return concat_embeddings(self.arg, embedding_module) * -1
