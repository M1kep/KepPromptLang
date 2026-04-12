from typing import List

from lark import Token, Transformer

from comfy.sd1_clip import SDTokenizer

from ..actions.base import Action, ActionArity
from .prompt_segment import PromptSegment
from .registration import get_action_by_name
from .utils import build_prompt_segment


class PromptTransformer(Transformer):
    """Maps the Lark parse tree into a flat list of PromptSegments and Actions."""

    def __init__(self, tokenizer: SDTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def item(self, items: List[Token]):
        for item in items:
            if isinstance(item, (Action, PromptSegment)):
                return item

            if item.type == "WORD":
                return build_prompt_segment(str(item), self.tokenizer)
            if item.type == "QUOTED_STRING":
                # Strip surrounding quotes, unescape \" and \'.
                unquoted = item[1:-1]
                unescaped = unquoted.replace('\\"', '"').replace("\\'", "'")
                return build_prompt_segment(unescaped, self.tokenizer)
            raise ValueError(f"Unknown item type: {item.type}")

    def arg(self, items):
        return items

    def embedding(self, items):
        return build_prompt_segment(
            f"{self.tokenizer.embedding_identifier}{items[0]}",
            self.tokenizer,
        )

    def generic_function(self, items):
        action = get_action_by_name(items[0])
        if action.arity == ActionArity.SINGLE:
            if len(items) != 2:
                raise ValueError(f"Action {action.action_name} expects exactly one argument")
            return action(items[1])
        if action.arity == ActionArity.MULTI:
            return action(items[1:])
        raise ValueError(f"Unknown action arity: {action.arity}")
