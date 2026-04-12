from typing import Tuple

from torch import Tensor
from torch.nn import Embedding

from .action_utils import concat_embeddings, get_total_length
from .base import PostModifiers, SingleArgAction


class PostPosAction(SingleArgAction):
    grammar = 'postPos(" arg+ ")"'

    display_name = "Ignore Positional Embeddings"
    action_name = "postPos"
    description = "Prevents positional embeddings from being applied to the provided segments or actions."
    usage_examples = [
        "A postPos(cat) on a rainy day",
    ]

    def token_length(self) -> int:
        return get_total_length(self.arg)

    def get_result(self, embedding_module: Embedding) -> Tuple[Tensor, PostModifiers]:
        return concat_embeddings(self.arg, embedding_module), PostModifiers(bypass_pos_embed=True)
