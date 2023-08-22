from typing import Callable, Union

from custom_nodes.ClipStuff.lib.actions.base import Action, PromptSegment


class ArithAction(Action):
    START_CHAR = "<"
    END_CHAR = ">"

    def __init__(self, base_segment: PromptSegment, ops: dict[str, list[Union[str, Action]]]):
        self.base_segment = base_segment
        self.ops = ops

    def __repr__(self):
        return f"ArithAction(\n\tbase_segment={self.base_segment},\n\tops={self.ops}\n)"

    def depth_repr(self, depth=1):
        out = "ArithAction(\n"
        if isinstance(self.base_segment, Action):
            base_segment_repr = self.base_segment.depth_repr(depth + 1)
            out += "\t" * depth + f"base_segment={base_segment_repr}\n"
        else:
            out += "\t" * depth + f'base_segment="{self.base_segment}",'
        for op_key, ops in self.ops.items():
            for op in ops:
                out += "\n" + "\t" * depth + f'"{op_key}":[\n'
                if isinstance(op, Action):
                    op_repr = op.depth_repr(depth + 2)
                    out += "\t" * (depth + 1) + f"{op_repr}\n"
                else:
                    out += "\t" * (depth + 1) + f'"{op}",\n'
                out += "\t" * depth + "],"
        out += "\n" + "\t" * (depth - 1) + ")"
        return out


    @classmethod
    def parse_segment(
        cls,
        tokens: list[str],
        start_chars: list[str],
        end_chars: list[str],
        parent_parser: Callable[[list[str]], Union[str, 'Action']],
    ) -> Action:
        """
        Parse an arithmetic action from a list of tokens
        Supported formats:
        <base_segment:+op1-op2-op3>

        :param tokens: List of tokens, will be modified
        :param start_chars: List of start chars for all actions
        :param end_chars: List of end chars for all actions
        :param parent_parser: Function to parse segments to allow for nested actions
        :return:
        """
        token = tokens.pop(0)
        assert token == cls.START_CHAR, "ArithAction must start with " + cls.START_CHAR + " but got " + token

        # Parse base segment
        base_segment = parent_parser(tokens)

        token = tokens.pop(0)
        assert token == ":", "ArithAction must have a ':' after the base segment" + " but got " + token

        # Parse ops string
        ops = {'+': [], '-': []}
        while tokens[0] != cls.END_CHAR:
            op_char = tokens.pop(0)
            assert op_char in ["+", "-"], "ArithAction must have a '+' or '-' as an op char but got " + op_char
            ops[op_char].append(parent_parser(tokens))

        token = tokens.pop(0)
        assert token == cls.END_CHAR, "ArithAction must end with " + cls.END_CHAR + " but got " + token

        return cls(base_segment, ops)
