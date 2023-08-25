from typing import Union

from custom_nodes.ClipStuff.lib.parser.prompt_segment import PromptSegment
from custom_nodes.ClipStuff.lib.action.base import Action

SegOrAction = Union[PromptSegment, Action]
