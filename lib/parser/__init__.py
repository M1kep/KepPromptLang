from lark import Lark

from .grammar import grammar

PromptParser = Lark(grammar, start="start", parser="earley")
