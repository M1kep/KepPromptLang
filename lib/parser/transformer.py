from lark import Transformer, Token

from comfy.sd1_clip import SD1Tokenizer
from custom_nodes.ClipStuff.lib.action.base import Action
from custom_nodes.ClipStuff.lib.actions.neg import NegAction
from custom_nodes.ClipStuff.lib.actions.norm import NormAction
from custom_nodes.ClipStuff.lib.actions.sum import SumAction
from custom_nodes.ClipStuff.lib.parser.prompt_segment import PromptSegment


class PromptTransformer(Transformer):
    # def WORD(self, items):
    #     return items

    def __init__(self, tokenizer: SD1Tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def item(self, items: list[Token]):
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

    def embedding(self, items):
        return build_prompt_segment(f'{self.tokenizer.embedding_identifier}{items[0]}', self.tokenizer)

    def function(self, items):
        for item in items:
            if item.data == 'sum_function':
                return SumAction(item.children[0], item.children[1:])
            elif item.data == 'neg_function':
                return NegAction(item.children)
            elif item.data == 'norm_function':
                return NormAction(item.children)
            else:
                raise Exception("Unknown function type: " + str(item.data))
