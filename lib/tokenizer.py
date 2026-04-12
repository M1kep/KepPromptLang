"""DSL-aware tokenizers.

Override `tokenize_with_weights` to parse our DSL and emit ComfyUI's native
`List[List[(token, weight)]]` format, where `token` is an int id, an inline
TI tensor, or a lazily-evaluated `Action` instance. The downstream
`PromptLangSDClipModel.process_tokens` resolves Actions to tensors at encode
time (when the embedding module is available) and otherwise defers to comfy's
stock `process_tokens` for batching, masking, and embedding lookup.
"""

from typing import Dict, List, Tuple, Union

from lark import Tree

from comfy.sd1_clip import SD1Tokenizer, SDTokenizer

from .actions.base import Action
from .parser import PromptParser
from .parser.prompt_segment import PromptSegment
from .parser.transformer import PromptTransformer

# Side-effect import: registers all built-in actions with the parser.
from . import actions  # noqa: F401

TokenEntry = Tuple[Union[int, "Action", object], float]


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
        # Tracks how many *post-splice* slots `current` will occupy: int/tensor
        # entries count as 1, Action entries count as `token_length()`. The row
        # itself is shorter — `process_tokens` expands actions to fill the gap.
        used = 1

        def close(row: List[TokenEntry], used_slots: int) -> None:
            row.append((self.end_token, 1.0))
            row.extend([(pad_token, 1.0)] * (self.max_length - used_slots - 1))
            batches.append(row)

        for item in items:
            length = item.token_length()
            if used + length > self.max_length - 1:
                close(current, used)
                current = [(self.start_token, 1.0)]
                used = 1

            if isinstance(item, Action):
                current.append((item, 1.0))
            else:
                assert isinstance(item, PromptSegment)
                current.extend((tok, 1.0) for tok in item.tokens)
            used += length

        close(current, used)
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
