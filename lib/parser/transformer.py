from typing import List

from lark import Token, Transformer

from comfy.sd1_clip import SDTokenizer

from ..actions.action_utils import parse_numeric_arg
from ..actions.base import Action, ActionArity
from ..actions.weighted import WeightedGroup
from .prompt_segment import PromptSegment
from .registration import get_action_by_name
from .utils import build_prompt_segment


class PromptTransformer(Transformer):
    """Maps the Lark parse tree into a flat list of PromptSegments and Actions."""

    def __init__(self, tokenizer: SDTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.vars: dict = {}

    def assign(self, items):
        name = str(items[0])
        if name in self.vars:
            raise ValueError(f"Variable ${name} is already defined")
        self.vars[name] = items[1]
        return None

    def ref(self, items):
        name = str(items[0])
        if name not in self.vars:
            raise ValueError(f"Variable ${name} referenced before assignment")
        # Weight 1.0 makes the group transparent: _flatten and embedding_tensor
        # already recurse through WeightedGroup, so no new container type needed.
        return WeightedGroup(self.vars[name], weight=1.0)

    def item(self, items: List[Token]):
        for item in items:
            if isinstance(item, (Action, PromptSegment, WeightedGroup)):
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

    def weighted(self, items):
        arg_items, weight_token = items
        return WeightedGroup(arg_items, float(weight_token))

    def embedding(self, items):
        return build_prompt_segment(
            f"{self.tokenizer.embedding_identifier}{items[0]}",
            self.tokenizer,
        )

    def generic_function(self, items):
        # `emph(text|w)` is sugar for `(text:w)`; handled here so it doesn't need
        # to fit the Action ABC (it changes weights, not embeddings).
        if str(items[0]) == "emph":
            if len(items) != 3:
                raise ValueError("emph expects exactly two arguments: emph(text|weight)")
            weight = parse_numeric_arg(items[2], action_name="emph", role="weight", cast=float)
            return WeightedGroup(items[1], weight)

        action = get_action_by_name(items[0])
        if action.arity == ActionArity.SINGLE:
            if len(items) != 2:
                raise ValueError(f"Action {action.action_name} expects exactly one argument")
            return action(items[1])
        if action.arity == ActionArity.MULTI:
            return action(items[1:])
        raise ValueError(f"Unknown action arity: {action.arity}")
