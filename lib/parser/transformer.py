from typing import List

from lark import Transformer, Token

from comfy.sd1_clip import SD1Tokenizer, SDTokenizer
from custom_nodes.KepPromptLang.lib.action.base import Action, ActionArity
from custom_nodes.KepPromptLang.lib.actions.diff import DiffAction
from custom_nodes.KepPromptLang.lib.actions.rand import RandAction
from custom_nodes.KepPromptLang.lib.parser.registration import get_action_by_name
from custom_nodes.KepPromptLang.lib.parser.utils import build_prompt_segment
from custom_nodes.KepPromptLang.lib.actions.neg import NegAction
from custom_nodes.KepPromptLang.lib.actions.norm import NormAction
from custom_nodes.KepPromptLang.lib.actions.sum import SumAction
from custom_nodes.KepPromptLang.lib.parser.prompt_segment import PromptSegment


class PromptTransformer(Transformer):
    # def WORD(self, items):
    #     return items

    def __init__(self, tokenizer: SDTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def item(self, items: List[Token]):
        for item in items:
            if isinstance(item, Action):
                return item

            if isinstance(item, PromptSegment):
                return item

            if item.type == "WORD":
                return build_prompt_segment(str(item), self.tokenizer)
            elif item.type == "QUOTED_STRING":
                # Remove the quotes
                unquoted = item[1:-1]
                # Replace escaped quotes with quotes
                unescaped = unquoted.replace("\\\"", "\"").replace("\\\'", "\'")
                return build_prompt_segment(unescaped, self.tokenizer)
            elif item.type == "embedding":
                return build_prompt_segment(item, self.tokenizer)
            elif item.type == "function":
                return item
            else:
                raise Exception("Unknown item type: " + str(item.type))

    def arg(self, items):
        return items

    def embedding(self, items):
        return build_prompt_segment(f'{self.tokenizer.embedding_identifier}{items[0]}', self.tokenizer)

    def generic_function(self, items):
        action = get_action_by_name(items[0])
        if action.arity == ActionArity.SINGLE:
            if len(items) != 2:
                raise ValueError(f"Action {action.name} should have exactly one argument")
            return action(items[1])
        elif action.arity == ActionArity.MULTI:
            return action(items[1:][:])
        else:
            raise ValueError(f"Unknown action arity: {action.arity}")
