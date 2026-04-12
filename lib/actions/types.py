from typing import Union

from ..parser.prompt_segment import PromptSegment
from .base import Action
from .weighted import WeightedGroup

SegOrAction = Union[PromptSegment, Action, WeightedGroup]
