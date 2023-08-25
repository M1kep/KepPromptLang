from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Embedding

from custom_nodes.ClipStuff.lib.parser.prompt_segment import PromptSegment


class Action(ABC):
    @property
    @abstractmethod
    def START_CHAR(self):
        pass

    @property
    @abstractmethod
    def END_CHAR(self):
        pass

    @abstractmethod
    def token_length(self):
        pass

    @abstractmethod
    def get_all_segments(self) -> list[PromptSegment]:
        pass

    @abstractmethod
    def get_result(self, embedding_module: Embedding) -> Tensor:
        pass

    def depth_repr(self, depth=1):
        raise NotImplementedError()


class SingleTargetAction(Action):
    def token_length(self):
        total_length = 0
        for seg_or_action in self.target:
            total_length += seg_or_action.token_length()

        return total_length

    def get_all_segments(self):
        segments: list[PromptSegment] = []
        for seg_or_action in self.target:
            if isinstance(seg_or_action, Action):
                segments.extend(seg_or_action.get_all_segments())
            else:
                segments.append(seg_or_action)
        return segments

    START_CHAR = "["
    END_CHAR = "]"

    def __init__(self, target: list[PromptSegment | Action]):
        # TODO: Target is a list now... what does this mean for us..
        self.target = target

    def __repr__(self):
        return f"neg({self.target})"
