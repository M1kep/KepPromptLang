from custom_nodes.ClipStuff.lib.actions.types import SegOrAction

def batch_size_info(batch: list[SegOrAction]):
    for segment in batch:
        print("Token Len: " + str(segment.token_length()))
        print(segment.depth_repr())
