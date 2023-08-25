from typing import Optional, Callable

import torch
from torch.nn import Embedding

from comfy.sd1_clip import SD1Tokenizer
from custom_nodes.ClipStuff.lib.action.base import Action
from .types import SegOrAction


class NudgeAction(Action):
    def token_length(self):
        # Nudge nudges the embeddings of the base segment, so the length is the length of the base segment
        if isinstance(self.base_segment, Action):
            return self.base_segment.token_length()

        return len(self.base_segment.tokens)

    def get_all_segments(self):
        segments = []
        if isinstance(self.base_segment, Action):
            segments += self.base_segment.get_all_segments()
        else:
            segments.append(self.base_segment)

        if isinstance(self.target, Action):
            segments += self.target.get_all_segments()
        else:
            segments.append(self.target)

        return segments

    def get_result(self, embedding_module: Embedding):
        if isinstance(self.base_segment, Action):
            base_segment_result = self.base_segment.get_result(embedding_module)
        else:
            base_segment_result = self.base_segment.get_embeddings(embedding_module)

        if isinstance(self.target, Action):
            target_segment_result = self.target.get_result(embedding_module)
        else:
            target_segment_result = self.target.get_embeddings(embedding_module)

        base_mean = torch.mean(base_segment_result, dim=1, keepdim=True)
        if target_segment_result.shape[1] == 1:
            translation_vector = target_segment_result - base_mean
        else:
            translation_vector = torch.mean(target_segment_result, dim=1, keepdim=True) - base_mean

        return base_segment_result.add(translation_vector, alpha=self.weight)

    START_CHAR = "["
    END_CHAR = "]"

    def __init__(
        self,
        base_segment: SegOrAction,
        target: SegOrAction,
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
        parent_parser: Callable[[list[str], SD1Tokenizer], SegOrAction],
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
