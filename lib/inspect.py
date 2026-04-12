"""Debug helper: report what a DSL prompt resolves to at the embedding layer."""

from typing import List, Tuple

import torch

from .actions.base import ACTION_CONTINUATION, Action


def inspect_prompt(clip, text: str, top_k: int = 3) -> str:
    """Tokenize + resolve actions and report per-slot L2 norm and nearest vocab tokens.

    Runs only the embedding lookup (no transformer forward), so it's cheap.
    """
    inner_clip, inner_tok = _unwrap(clip)
    embedding_module = inner_clip.transformer.get_input_embeddings()
    weight = embedding_module.weight.to(torch.float32)
    weight_norm = torch.nn.functional.normalize(weight, dim=-1)

    batches = inner_tok.tokenize_with_weights(text)

    lines = [f"Prompt: {text!r}", ""]
    for batch_idx, batch in enumerate(batches):
        lines.append(f"-- batch {batch_idx} ({len(batch)} entries) --")
        lines.append(f"{'idx':>3}  {'w':>5}  {'src':<24}  {'L2':>6}  nearest")
        position = 0
        for token, w in batch:
            if token is ACTION_CONTINUATION:
                position += 1
                continue
            embeds, source = _resolve(token, embedding_module)
            for row in embeds:
                norm = torch.norm(row).item()
                nearest = _nearest_vocab(row, weight_norm, inner_tok, top_k)
                lines.append(f"{position:>3}  {w:>5.2f}  {source:<24.24}  {norm:>6.3f}  {nearest}")
                position += 1
        lines.append("")

    return "\n".join(lines)


def _unwrap(clip):
    """Dig past SD1ClipModel/SDXL wrappers to the underlying SDClipModel + SDTokenizer."""
    cond = clip.cond_stage_model
    tok = clip.tokenizer
    inner_clip = getattr(cond, getattr(cond, "clip", "clip_l"), cond)
    inner_tok = getattr(tok, getattr(tok, "clip", "clip_l"), tok)
    return inner_clip, inner_tok


def _resolve(token, embedding_module) -> Tuple[torch.Tensor, str]:
    """Map a token entry to its `[N, hidden]` embedding rows and a short source label."""
    if isinstance(token, Action):
        result = token.get_result(embedding_module)
        tensor = result[0] if isinstance(result, tuple) else result
        return tensor.reshape(-1, tensor.shape[-1]).to(torch.float32), repr(token)
    if isinstance(token, int):
        return embedding_module.weight[token : token + 1].to(torch.float32), f"tok#{token}"
    # Inline TI tensor.
    return token.reshape(-1, token.shape[-1]).to(torch.float32), "embedding:"


def _nearest_vocab(row: torch.Tensor, weight_norm: torch.Tensor, tokenizer, top_k: int) -> str:
    row_norm = torch.nn.functional.normalize(row.unsqueeze(0), dim=-1)
    sims = (row_norm @ weight_norm.T).squeeze(0)
    top_ids: List[int] = sims.topk(top_k).indices.tolist()
    inv_vocab = getattr(tokenizer, "inv_vocab", {})
    return ", ".join(inv_vocab.get(tid, f"#{tid}") for tid in top_ids)
