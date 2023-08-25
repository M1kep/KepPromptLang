from typing import Union

from custom_nodes.ClipStuff.lib.prompt_segment import PromptSegment
from .base import Action

SegOrAction = Union[PromptSegment, Action]
