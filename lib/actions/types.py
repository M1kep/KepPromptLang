from typing import Union

from ..parser.prompt_segment import PromptSegment
from .base import Action

SegOrAction = Union[PromptSegment, Action]
