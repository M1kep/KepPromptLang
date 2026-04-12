from typing import Callable, List, TypeVar

import torch
from torch import Tensor
from torch.nn import Embedding

from .base import Action
from .types import SegOrAction


def embedding_tensor(seg_or_action: SegOrAction, embedding_module: Embedding) -> Tensor:
    """Embeddings for a segment, or get_result() for an action — always a bare tensor."""
    if isinstance(seg_or_action, Action):
        result = seg_or_action.get_result(embedding_module)
        return result[0] if isinstance(result, tuple) else result
    return seg_or_action.get_embeddings(embedding_module)


def get_total_length(args: List[SegOrAction]) -> int:
    return sum(seg_or_action.token_length() for seg_or_action in args)


def concat_embeddings(args: List[SegOrAction], embedding_module: Embedding) -> Tensor:
    """Materialize and concatenate embeddings for a sequence of segments/actions along the seq dim."""
    return torch.cat([embedding_tensor(x, embedding_module) for x in args], dim=1)


def add_with_broadcast(result: Tensor, arg_embedding: Tensor, op: str) -> Tensor:
    """Add or subtract arg_embedding into result, averaging arg over the seq dim if shapes mismatch."""
    matched = arg_embedding.shape[-2] == 1 or result.shape[-2] == arg_embedding.shape[-2]
    if not matched:
        print(f"WARNING: shape mismatch when trying to apply {op}, arg will be averaged")
        arg_embedding = torch.mean(arg_embedding, dim=1, keepdim=True)
    return result.add(arg_embedding) if op == "add" else result.sub(arg_embedding)


T = TypeVar("T")


def parse_numeric_arg(
    arg: List[SegOrAction],
    *,
    action_name: str,
    role: str,
    cast: Callable[[str], T] = float,
) -> T:
    """Pull a single numeric value out of a one-segment arg, with helpful errors.

    Used by every action that takes a scalar weight/multiplier/length.
    """
    if len(arg) != 1:
        raise ValueError(f"{action_name} {role} should have exactly one segment")
    item = arg[0]
    if isinstance(item, Action):
        raise ValueError(f"{action_name} {role} cannot be an action")
    try:
        return cast(item.text)
    except ValueError:
        raise ValueError(f"{action_name} {role} should be a {cast.__name__}")
