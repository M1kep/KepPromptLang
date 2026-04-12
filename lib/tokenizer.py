"""DSL-aware tokenizers.

Override `tokenize_with_weights` to return our `List[List[SegOrAction]]` instead of
the standard `[(token_id, weight), ...]` shape. The downstream `PromptLangSDClipModel`
knows how to consume this.
"""

from typing import Dict, List

from lark import Tree

from comfy.sd1_clip import SD1Tokenizer, SDTokenizer

from .actions.types import SegOrAction
from .parser import PromptParser
from .parser.prompt_segment import PromptSegment
from .parser.transformer import PromptTransformer

# Side-effect import: registers all built-in actions with the parser.
from . import actions  # noqa: F401


class PromptLangSDTokenizer(SDTokenizer):
    """Returns batches of segments/actions instead of (token, weight) pairs."""

    def tokenize_with_weights(  # type: ignore[override]
        self, text: str, return_word_ids: bool = False, **kwargs
    ) -> List[List[SegOrAction]]:
        pad_token = self.end_token if self.pad_with_end else 0

        parsed_prompt = PromptParser.parse(text)
        parsed = PromptTransformer(self).transform(parsed_prompt)
        items = parsed.children if isinstance(parsed, Tree) else [parsed]

        batches: List[List[SegOrAction]] = []
        current: List[SegOrAction] = [PromptSegment(text="[SOT]", tokens=[self.start_token])]
        current_size = 1

        for segment in items:
            num_tokens = segment.token_length()

            if num_tokens + current_size > self.max_length - 1:
                remaining = self.max_length - current_size
                current.append(_pad_segment(self.end_token, pad_token, remaining))
                batches.append(current)

                current = [PromptSegment(text="[SOT]", tokens=[self.start_token]), segment]
                current_size = num_tokens + 1
            else:
                current.append(segment)
                current_size += num_tokens

        remaining = self.max_length - current_size
        current.append(_pad_segment(self.end_token, pad_token, remaining))
        batches.append(current)

        return batches


def _pad_segment(end_token: int, pad_token: int, remaining: int) -> PromptSegment:
    """Build a trailing [EOT] + pad_token * (remaining-1) segment."""
    return PromptSegment("__PAD__", [end_token] + [pad_token] * (remaining - 1))


class PromptLangSD1Tokenizer(SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None, clip_name="l", tokenizer=PromptLangSDTokenizer):
        super().__init__(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data or {},
            clip_name=clip_name,
            tokenizer=tokenizer,
        )


class PromptLangSDXLClipGTokenizer(PromptLangSDTokenizer):
    def __init__(self, tokenizer_path=None, embedding_directory=None, tokenizer_data=None):
        super().__init__(
            tokenizer_path=tokenizer_path,
            pad_with_end=False,
            embedding_directory=embedding_directory,
            embedding_size=1280,
            embedding_key="clip_g",
            tokenizer_data=tokenizer_data or {},
        )


class PromptLangSDXLTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data=None) -> None:
        td = tokenizer_data or {}
        self.clip_l = PromptLangSDTokenizer(embedding_directory=embedding_directory, tokenizer_data=td)
        self.clip_g = PromptLangSDXLClipGTokenizer(embedding_directory=embedding_directory, tokenizer_data=td)

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False, **kwargs) -> Dict[str, List[List[SegOrAction]]]:
        return {
            "g": self.clip_g.tokenize_with_weights(text, return_word_ids, **kwargs),
            "l": self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs),
        }

    def untokenize(self, token_weight_pair):
        return self.clip_g.untokenize(token_weight_pair)

    def state_dict(self):
        return {}
