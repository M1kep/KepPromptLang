from typing import Optional, Union, Callable

from comfy.sd1_clip import SD1Tokenizer
from custom_nodes.ClipStuff.lib.actions.base import Action, PromptSegment


class NudgeAction(Action):
    START_CHAR = "["
    END_CHAR = "]"

    def __init__(
        self,
        base_segment: PromptSegment | Action,
        target: Union[PromptSegment, Action],
        weight: Optional[float] = None,
    ):
        self.base_segment = base_segment
        self.weight = weight
        self.target = target

    def __repr__(self):
        return f"NudgeAction(\n\tbase_segment={self.base_segment},\n\ttarget={self.target},\n\tweight={self.weight}\n)"

    def depth_repr(self, depth=1):
        out = "NudgeAction(\n"
        if isinstance(self.base_segment, Action):
            base_segment_repr = self.base_segment.depth_repr(depth + 1)
            out += "\t" * depth + f"base_segment={base_segment_repr}\n"
        else:
            out += "\t" * depth + f'base_segment={self.base_segment.depth_repr()},\n'

        if isinstance(self.target, Action):
            target_repr = self.target.depth_repr(depth + 1)
            out += "\t" * depth + f"target={target_repr},\n"
        else:
            out += "\t" * depth + f"target={self.target.depth_repr()},\n"
        out += "\t" * depth + f"weight={self.weight},\n"
        out += "\t" * (depth - 1) + ")"
        return out

    @classmethod
    def parse_segment(
        cls,
        tokens: list[str],
        start_chars: list[str],
        end_chars: list[str],
        parent_parser: Callable[[list[str], SD1Tokenizer], PromptSegment | Action],
        tokenizer: SD1Tokenizer,
    ) -> Action:
        """
        Parse a nudge action from a list of tokens
        Supported formats:
        [base_segment:target_segment]
        [base_segment:target_segment:weight]

        Weight is optional, if not provided it will be None
        :param tokens: List of tokens, will be modified
        :param start_chars: List of start chars for all actions
        :param end_chars: List of end chars for all actions
        :param parent_parser: Function to parse segments to allow for nested actions
        :return:
        """
        token = tokens.pop(0)
        assert token == cls.START_CHAR, "NudgeAction must start with " + cls.START_CHAR + " got " + token

        # Parse base segment
        base_segment = parent_parser(tokens, tokenizer)

        token = tokens.pop(0)
        assert token == ":", "NudgeAction must have a ':' after the base segment" + " but got " + token

        # Parse target segment
        target_segment = parent_parser(tokens, tokenizer)

        # Parse weight if it exists
        weight = None
        if tokens[0] == ":":
            # Parse weight
            tokens.pop(0)
            weight = float(tokens.pop(0))

        token = tokens.pop(0)
        assert token == cls.END_CHAR, "NudgeAction must end with " + cls.END_CHAR + " got " + token

        return cls(base_segment, target_segment, weight)
