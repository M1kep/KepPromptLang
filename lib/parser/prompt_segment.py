from typing import List, Union

import torch
from torch import Tensor
from torch.nn import Embedding


class PromptSegment:
    """A run of contiguous tokens from the user's prompt, possibly with inline TI tensor entries."""

    def __init__(self, text: str, tokens: List[Union[int, Tensor]]):
        self.text = text
        self.tokens = tokens

    def __repr__(self) -> str:
        cleaned = ", ".join(str(t) if isinstance(t, int) else "EMBD" for t in self.tokens)
        return f'"{self.text}"({cleaned})'

    def token_length(self) -> int:
        return len(self.tokens)

    def get_embeddings(self, embedding_module: Embedding) -> Tensor:
        """Look up embeddings for plain int tokens.

        Inline TI tensors aren't handled here; the encoder splices them in at a higher level.
        """
        ids = torch.LongTensor([t for t in self.tokens if isinstance(t, int)]).to(
            embedding_module.weight.device
        )
        return embedding_module(ids.unsqueeze(0))
