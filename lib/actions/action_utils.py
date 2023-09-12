from typing import List

import torch
from torch import Tensor
from torch.nn import Embedding

from custom_nodes.KepPromptLang.lib.action.base import Action
from custom_nodes.KepPromptLang.lib.actions.types import SegOrAction


def get_embedding(seg_or_action: SegOrAction, embedding_module: Embedding) -> Tensor:
    if isinstance(seg_or_action, Action):
        return seg_or_action.get_result(embedding_module)
    return seg_or_action.get_embeddings(embedding_module)


def get_embedding_for_segments(
    segments: List[SegOrAction], embedding_module: Embedding
) -> Tensor:
    return torch.cat([get_embedding(segment, embedding_module) for segment in segments], dim=1)
