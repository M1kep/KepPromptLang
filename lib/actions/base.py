from abc import ABC, abstractmethod
from typing import Callable, Union


class Action(ABC):
    @property
    @abstractmethod
    def START_CHAR(self):
        pass

    @property
    @abstractmethod
    def END_CHAR(self):
        pass

    @classmethod
    @abstractmethod
    def parse_segment(
        cls,
        tokens: list[str],
        start_chars: list[str],
        end_chars: list[str],
        parent_parser: Callable[[list[str]], Union[str, 'Action']],
    ) -> 'Action':
        pass

    def depth_repr(self, depth=1):
        raise NotImplementedError()

PromptSegment = str | Action
