from typing import List

from custom_nodes.KepPromptLang.lib.actions.types import SegOrAction

def batch_size_info(batch: List[SegOrAction]):
    for segment in batch:
        print("Token Len: " + str(segment.token_length()))
        print(segment.depth_repr())
