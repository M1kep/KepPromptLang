from abc import ABC, abstractmethod
from typing import Callable, Union

import torch
from torch import Tensor
from torch.nn import Embedding

from comfy.sd1_clip import SD1Tokenizer


class Action(ABC):
    @property
    @abstractmethod
    def START_CHAR(self):
        pass

    @property
    @abstractmethod
    def END_CHAR(self):
        pass

    @abstractmethod
    def token_length(self):
        pass

    @abstractmethod
    def get_all_segments(self):
        pass

    @abstractmethod
    def get_result(self, embedding_module: Embedding):
        pass

    @classmethod
    @abstractmethod
    def parse_segment(
        cls,
        tokens: list[str],
        start_chars: list[str],
        end_chars: list[str],
        parent_parser: Callable[[list[str], SD1Tokenizer], Union[str, 'Action']],
        tokenizer: SD1Tokenizer,
    ) -> 'Action':
        pass

    def depth_repr(self, depth=1):
        raise NotImplementedError()

class PromptSegment:
    def __init__(self, text: str, tokens: list[Union[int, Tensor]]):
        self.text = text
        self.tokens = tokens

    def token_length(self):
        return len(self.tokens)

    def get_embeddings(self, embedding_module: Embedding):
        tensors = torch.LongTensor(self.tokens).to(torch.device('cpu'))
        unsqueezed_tensors = tensors.unsqueeze(0)
        return embedding_module(unsqueezed_tensors)

    def depth_repr(self, depth=1):
        out = f'"{self.text}"('

        cleaned_tokens = list(map(lambda x: str(x) if isinstance(x, int) else "EMBD", self.tokens))
        out += ", ".join(cleaned_tokens)

        out += ")"
        return out

def build_prompt_segment(text: str, tokenizer: SD1Tokenizer) -> PromptSegment:
    split_text = text.split(" ")
    tokens = []
    for word in split_text:
        if word.startswith(tokenizer.embedding_identifier) and tokenizer.embedding_directory is not None:
            embedding_name = word[len(tokenizer.embedding_identifier):].strip('\n')

            get_embed_ret = tokenizer._try_get_embedding(embedding_name)
            embedding = get_embed_ret[0]
            leftover = get_embed_ret[1]
            if embedding is None:
                print(f"warning, embedding:{embedding_name} does not exist, ignoring")
            else:
                if len(embedding.shape) == 1:
                    tokens.append(embedding)
                else:
                    tokens.extend(embedding)

            if leftover != "":
                word = leftover
            else:
                continue
        tokens.extend(tokenizer.tokenizer(word)["input_ids"][1:-1])

    return PromptSegment(text, tokens)
