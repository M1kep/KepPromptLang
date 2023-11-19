from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Embedding

from custom_nodes.KepPromptLang.lib.action.base import (
    SingleArgAction,
    Action,
    PostModifiers,
)


class PostPosAction(SingleArgAction):
    grammar = 'postPos(" arg+ ")"'
    chars = ["[", "]"]

    display_name = "Ignore Positional Embeddings"
    action_name = "postPos"
    description = "Prevents positional embeddings from being applied to the provided segments or actions."
    usage_examples = [
        "A postPos(cat) on a rainy day",
    ]

    def __init__(self, args):
        super().__init__(args)
        self.result = None

    def token_length(self) -> int:
        """
        PostPos returns the results of the wrapped action, so the length is the length of the wrapped action
        """
        total_length = 0
        for seg_or_action in self.arg:
            total_length += seg_or_action.token_length()

        return total_length

    def get_result(self, embedding_module: Embedding) -> Tuple[Tensor, PostModifiers]:
        all_embeddings = []
        for seg_or_action in self.arg:
            if isinstance(seg_or_action, Action):
                all_embeddings.append(seg_or_action.get_result(embedding_module))
            else:
                all_embeddings.append(seg_or_action.get_embeddings(embedding_module))

        return torch.cat(all_embeddings, dim=1), PostModifiers(position_embed_scale=None, bypass_pos_embed=True)
