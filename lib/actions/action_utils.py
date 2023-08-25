from torch import Tensor
from torch.nn import Embedding

from custom_nodes.ClipStuff.lib.action.base import Action
from custom_nodes.ClipStuff.lib.actions.types import SegOrAction


def get_embedding(seg_or_action: SegOrAction, embedding_module: Embedding) -> Tensor:
    if isinstance(seg_or_action, Action):
        return seg_or_action.get_result(embedding_module)
    return seg_or_action.get_embeddings(embedding_module)
