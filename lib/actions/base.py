from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Embedding

from ..parser.prompt_segment import PromptSegment


class ActionArity(Enum):
    NONE = 0
    SINGLE = 1
    MULTI = 2


@dataclass
class PostModifiers:
    """Optional position-embedding tweaks an action can request for its token range.

    `start_idx` / `end_idx` are filled in by the encoder once the action's position in
    the final token stream is known.
    """
    position_embed_scale: Optional[float] = None
    bypass_pos_embed: bool = False
    start_idx: int = 0
    end_idx: int = 0


ActionResult = Union[Tensor, Tuple[Tensor, PostModifiers]]

# Tokenizer placeholder for the 2nd..Nth slots of a multi-token Action, so each
# row stays exactly max_length entries (required for comfy's per-position weight
# indexing). process_tokens drops these; the Action's tensor fills the slots.
ACTION_CONTINUATION = object()


class Action(ABC):
    arity: ActionArity = ActionArity.NONE
    display_name: str = ""
    action_name: str = ""
    description: str = ""
    grammar: str = ""
    usage_examples: List[str] = []

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def token_length(self) -> int: ...

    @abstractmethod
    def get_result(self, embedding_module: Embedding) -> ActionResult: ...


class SingleArgAction(Action, ABC):
    arity = ActionArity.SINGLE

    def __init__(self, arg: List[Union[PromptSegment, "Action"]]):
        self.arg = arg

    def __repr__(self) -> str:
        return f"{self.action_name}({self.arg})"


class MultiArgAction(Action, ABC):
    arity = ActionArity.MULTI

    def __init__(self, args: List[List[Union[PromptSegment, "Action"]]]):
        self.all_args = args

    def __repr__(self) -> str:
        joined = " | ".join(str(a) for a in self.all_args)
        return f"{self.action_name}({joined})"
