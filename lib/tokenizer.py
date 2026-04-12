"""DSL-aware tokenizers.

Override `tokenize_with_weights` to parse our DSL and emit ComfyUI's native
`List[List[(token, weight)]]` format, where `token` is an int id, an inline
TI tensor, a lazily-evaluated `Action`, or `ACTION_CONTINUATION`.

Row alignment matters: comfy's stock `encode_token_weights` indexes weights by
post-transformer position, so each row must be exactly `max_length` entries.
A multi-slot Action is therefore emitted as one `(action, w)` entry followed by
`(ACTION_CONTINUATION, w)` placeholders; `process_tokens` drops the placeholders
and the action's tensor expands to fill those slots.
"""

from typing import Dict, Iterable, List, Tuple, Union

from lark import Tree

from comfy.sd1_clip import SD1Tokenizer, SDTokenizer

from .actions.base import ACTION_CONTINUATION, Action
from .actions.weighted import WeightedGroup
from .parser import PromptParser
from .parser.prompt_segment import PromptSegment
from .parser.transformer import PromptTransformer

# Side-effect import: registers all built-in actions with the parser.
from . import actions  # noqa: F401

TokenEntry = Tuple[Union[int, "Action", object], float]


def _flatten(item, weight: float) -> Iterable[TokenEntry]:
    """Walk the parsed item tree, yielding one (token, weight) entry per output slot."""
    if isinstance(item, WeightedGroup):
        for sub in item.items:
            yield from _flatten(sub, weight * item.weight)
    elif isinstance(item, Action):
        yield (item, weight)
        for _ in range(item.token_length() - 1):
            yield (ACTION_CONTINUATION, weight)
    elif isinstance(item, PromptSegment):
        for tok in item.tokens:
            yield (tok, weight)
    else:
        raise TypeError(f"Unexpected parse item {item!r} ({type(item).__name__})")


class PromptLangSDTokenizer(SDTokenizer):
    def tokenize_with_weights(  # type: ignore[override]
        self, text: str, return_word_ids: bool = False, **kwargs
    ) -> List[List[TokenEntry]]:
        # SDXL passes a pre-parsed tree to avoid re-running Lark per sub-tokenizer.
        tree = kwargs.pop("_parsed_tree", None) or PromptParser.parse(text)
        return self._batch_from_tree(tree)

    def _batch_from_tree(self, tree) -> List[List[TokenEntry]]:
        pad_token = self.end_token if self.pad_with_end else 0

        parsed = PromptTransformer(self).transform(tree)
        items = parsed.children if isinstance(parsed, Tree) else [parsed]

        batches: List[List[TokenEntry]] = []
        current: List[TokenEntry] = [(self.start_token, 1.0)]

        def close(row: List[TokenEntry]) -> None:
            row.append((self.end_token, 1.0))
            row.extend([(pad_token, 1.0)] * (self.max_length - len(row)))
            batches.append(row)

        for item in items:
            entries = list(_flatten(item, 1.0))
            if len(current) + len(entries) > self.max_length - 1:
                close(current)
                current = [(self.start_token, 1.0)]
            current.extend(entries)

        close(current)
        return batches


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

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False, **kwargs) -> Dict[str, List[List[TokenEntry]]]:
        tree = PromptParser.parse(text)
        return {
            "g": self.clip_g.tokenize_with_weights(text, return_word_ids, _parsed_tree=tree, **kwargs),
            "l": self.clip_l.tokenize_with_weights(text, return_word_ids, _parsed_tree=tree, **kwargs),
        }

    def untokenize(self, token_weight_pair):
        return self.clip_g.untokenize(token_weight_pair)

    def state_dict(self):
        return {}
