from typing import Union

from custom_nodes.KepPromptLang.lib.parser.prompt_segment import PromptSegment
from custom_nodes.KepPromptLang.lib.action.base import Action

SegOrAction = Union[PromptSegment, Action]
