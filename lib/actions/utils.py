import torch


def slerp(val: float, low: torch.Tensor, high: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Spherical linear interpolation between two tensors along the last dim."""
    val_t = torch.tensor(val, dtype=torch.float32, device=low.device).clamp(0, 1)

    low_norm = low / torch.norm(low, dim=-1, keepdim=True)
    high_norm = high / torch.norm(high, dim=-1, keepdim=True)

    dot = (low_norm * high_norm).sum(-1, keepdim=True).clamp(-1, 1)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    scale_low = torch.sin((1.0 - val_t) * omega) / (sin_omega + epsilon)
    scale_high = torch.sin(val_t * omega) / (sin_omega + epsilon)

    # Fall back to linear interp where the angle is too small for stable slerp.
    close = sin_omega < epsilon
    scale_low = torch.where(close, 1.0 - val_t, scale_low)
    scale_high = torch.where(close, val_t, scale_high)

    return scale_low * low + scale_high * high
