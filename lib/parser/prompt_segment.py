from typing import Union, List

import torch
from torch import Tensor
from torch.nn import Embedding


class PromptSegment:
    def __init__(self, text: str, tokens: List[Union[int, Tensor]]):
        self.text = text
        self.tokens = tokens

    def __repr__(self):
        return f'"{self.text}"{self.tokens}'

    def token_length(self):
        return len(self.tokens)

    def get_embeddings(self, embedding_module: Embedding) -> Tensor:
        # TODO: Get properly get the device...
        tensors = torch.LongTensor(self.tokens).to(torch.device('cuda'))
        unsqueezed_tensors = tensors.unsqueeze(0)
        return embedding_module(unsqueezed_tensors)

    def depth_repr(self, depth=1):
        out = f'"{self.text}"('

        cleaned_tokens = list(map(lambda x: str(x) if isinstance(x, int) else "EMBD", self.tokens))
        out += ", ".join(cleaned_tokens)

        out += ")"
        return out
