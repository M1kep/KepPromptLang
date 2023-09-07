from typing import List

import torch

from custom_nodes.KepPromptLang.lib.actions.types import SegOrAction

def batch_size_info(batch: List[SegOrAction]):
    for segment in batch:
        print("Token Len: " + str(segment.token_length()))
        print(segment.depth_repr())

def slerp(val: float, low: torch.Tensor, high: torch.Tensor, epsilon=1e-5):
    # Convert val to tensor and clamp between 0 and 1
    val = torch.tensor(val, dtype=torch.float32).clamp(0, 1)

    # Normalize the vectors
    low_norm = low / torch.norm(low, dim=-1, keepdim=True)
    high_norm = high / torch.norm(high, dim=-1, keepdim=True)

    # Calculate the cosine of the angle between the vectors
    dot = (low_norm * high_norm).sum(-1, keepdim=True)

    # Clamp to prevent numerical errors
    dot = torch.clamp(dot, -1, 1)

    omega = torch.acos(dot)

    # Slerp formula
    sin_omega = torch.sin(omega)
    scale_0 = torch.sin((1.0 - val) * omega) / (sin_omega + epsilon)
    scale_1 = torch.sin(val * omega) / (sin_omega + epsilon)

    # Handle the case where omega is small (the vectors are close)
    close_condition = sin_omega < epsilon
    scale_0 = torch.where(close_condition, 1.0 - val, scale_0)
    scale_1 = torch.where(close_condition, val, scale_1)

    return scale_0 * low + scale_1 * high

def is_broadcastable(tensor1, tensor2) -> bool:
    """
    Check if two tensors are broadcastable.

    Parameters:
    - tensor1 (torch.Tensor): The target tensor against which broadcastability of tensor2 is checked.
    - tensor2 (torch.Tensor): The tensor whose broadcastability is to be verified against tensor1.

    Returns:
    - bool: True if tensor2 is broadcastable to tensor1, False otherwise.
    """
    try:
        broadcasted_shape = torch.broadcast_shapes(tensor1.shape, tensor2.shape)
        return True
    except RuntimeError:
        return False
