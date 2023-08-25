from abc import ABC, abstractmethod
from typing import Union

from torch import Tensor
from torch.nn import Embedding

from custom_nodes.ClipStuff.lib.parser.prompt_segment import PromptSegment


class Action(ABC):
    @property
    @abstractmethod
    def chars(self) -> list[str] | None:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def grammar(self) -> str:
        """
        The grammar for this action. This is used to parse the action from the prompt.
        :return:
        """
        pass

    @abstractmethod
    def token_length(self) -> int:
        """
        The length of the tokens that this action will add to the prompt.
        :return:
        """
        pass

    @abstractmethod
    def get_all_segments(self) -> list[PromptSegment]:
        """
        Get all segments, including nested segments.
        :return:
        """
        pass

    @abstractmethod
    def get_result(self, embedding_module: Embedding) -> Tensor:
        """
        Get the result of this action. This is called when the embeddings are being calculated.
        :param embedding_module: The embedding module to use to get the base embeddings for tokens.
        :return:
        """
        pass

    def depth_repr(self, depth: int = 1) -> str:
        raise NotImplementedError()


class SingleArgAction(Action, ABC):
    def get_all_segments(self) -> list[PromptSegment]:
        segments = []
        for seg_or_action in self.arg:
            if isinstance(seg_or_action, Action):
                segments.extend(seg_or_action.get_all_segments())
            else:
                segments.append(seg_or_action)
        return segments

    def __init__(self, arg: list[PromptSegment | Action]):
        # TODO: Target is a list now... what does this mean for us..
        self.arg = arg

    def __repr__(self) -> str:
        return f"{self.name}({self.arg})"

class MultiArgAction(Action, ABC):
    def get_all_segments(self) -> list[PromptSegment]:
        segments = []
        for seg_or_action in self.base_segment:
            if isinstance(seg_or_action, Action):
                segments.extend(seg_or_action.get_all_segments())
            else:
                segments.append(seg_or_action)

        for arg in self.args:
            for seg_or_action in arg:
                if isinstance(seg_or_action, Action):
                    segments.extend(seg_or_action.get_all_segments())
                else:
                    segments.append(seg_or_action)

        return segments

    def __init__(
            self,
            base_segment: list[PromptSegment | Action],
            args: list[list[Union[PromptSegment, Action]]],
    ):
        self.base_segment = base_segment
        self.args = args
