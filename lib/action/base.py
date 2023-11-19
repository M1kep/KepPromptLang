from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, List, TypedDict, Tuple

from torch import Tensor
from torch.nn import Embedding
from transformers.models.clip.modeling_clip import CLIPTextTransformer

from custom_nodes.KepPromptLang.lib.parser.prompt_segment import PromptSegment


class ActionArity(Enum):
    NONE = 0
    SINGLE = 1
    MULTI = 2

class PostModifiers(TypedDict):
    """
    A dictionary of post modifiers for an action result.
    """
    position_embed_scale: Union[float, None]
    bypass_pos_embed: Union[bool, None]


class Action(ABC):
    @property
    @abstractmethod
    def chars(self) -> Union[List[str], None]:
        pass

    @property
    @abstractmethod
    def arity(self) -> ActionArity:
        """
        Determines the arity of the action. This is used to determine how many arguments the action supports.
        :return:
        """
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
    def get_all_segments(self) -> List[PromptSegment]:
        """
        Get all segments, including nested segments.
        :return:
        """
        pass

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the action. This is called when the action is parsed from the prompt.
        :param args: The arguments for the action.
        """
        pass

    @abstractmethod
    def get_result(self, embedding_module: Embedding) -> Union[Tensor, Tuple[Tensor, PostModifiers]]:
        """
        Get the result of this action. This is called when the embeddings are being calculated.
        :param embedding_module: The embedding module to use to get the base embeddings for tokens.
        :return:
        """
        pass

    def process_with_transformer(self, transformer: CLIPTextTransformer, embedding_module: Embedding) -> None:
        """
        For actions that need access to the TextTransformer, this method is called. Results are expected to be returned via get_result still.
        :param transformer: An instance of CLIPTextTransformer
        """
        pass

    def depth_repr(self, depth: int = 1) -> str:
        raise NotImplementedError()


class SingleArgAction(Action, ABC):
    arity = ActionArity.SINGLE
    def get_all_segments(self) -> List[PromptSegment]:
        segments = []
        for seg_or_action in self.arg:
            if isinstance(seg_or_action, Action):
                segments.extend(seg_or_action.get_all_segments())
            else:
                segments.append(seg_or_action)
        return segments

    def process_with_transformer(self, transformer: CLIPTextTransformer, embedding_module: Embedding) -> None:
        for seg_or_action in self.arg:
            if isinstance(seg_or_action, Action):
                seg_or_action.process_with_transformer(transformer, embedding_module)

    def __init__(self, arg: List[Union[PromptSegment, Action]]):
        # TODO: Target is a list now... what does this mean for us..
        self.arg = arg

    def __repr__(self) -> str:
        return f"{self.name}({self.arg})"

class MultiArgAction(Action, ABC):
    arity = ActionArity.MULTI
    def get_all_segments(self) -> List[PromptSegment]:
        segments = []
        for arg in self.all_args:
            for seg_or_action in arg:
                if isinstance(seg_or_action, Action):
                    segments.extend(seg_or_action.get_all_segments())
                else:
                    segments.append(seg_or_action)

        return segments

    def process_with_transformer(self, transformer: CLIPTextTransformer, embedding_module: Embedding) -> None:
        for arg in self.all_args:
            for seg_or_action in arg:
                if isinstance(seg_or_action, Action):
                    seg_or_action.process_with_transformer(transformer, embedding_module)

    def __init__(
            self,
            args: List[List[Union[PromptSegment, Action]]],
    ):
        self.all_args = args
