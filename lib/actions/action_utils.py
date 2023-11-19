from typing import List

from torch import Tensor
from torch.nn import Embedding

from custom_nodes.KepPromptLang.lib.action.base import Action
from custom_nodes.KepPromptLang.lib.actions.types import SegOrAction


def get_embedding(seg_or_action: SegOrAction, embedding_module: Embedding) -> Tensor:
    if isinstance(seg_or_action, Action):
        return seg_or_action.get_result(embedding_module)
    return seg_or_action.get_embeddings(embedding_module)

def get_total_length(args: List[SegOrAction]) -> int:
    total_length = 0
    for seg_or_action in args:
        total_length += seg_or_action.token_length()

    return total_length
