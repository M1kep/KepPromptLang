import re
from typing import Union

from comfy.sd1_clip import SD1Tokenizer
from custom_nodes.ClipStuff.lib.actions import (
    NudgeAction,
    ArithAction,
    ALL_START_CHARS,
    ALL_END_CHARS,
    ALL_ACTIONS,
)
from custom_nodes.ClipStuff.lib.actions.base import Action, PromptSegment
from custom_nodes.ClipStuff.lib.actions.lib import (
    is_any_action_segment,
    is_action_segment,
)


arith_action = r'(<[a-zA-Z0-9\-_]+:[a-zA-Z0-9\-_]+>)'

tokenizer_regex = re.compile(
    fr'\d+\.\d+|\d+|[\w\s]+|[:+-{re.escape("".join(ALL_START_CHARS))}{re.escape("".join(ALL_END_CHARS))}]'
)
def tokenize(text: str) -> list[str]:
    # Captures:
    # 1. Words
    # 2. Numbers(1.0, 1)
    # 3. Special characters(ALL_START_CHARS, ALL_END_CHARS, :, +, -)
    tokens = re.findall(tokenizer_regex, text)
    print(tokens)
    return [token.strip() for token in tokens]



def parse_segment(tokens: list[str], tokenizer: SD1Tokenizer) -> PromptSegment | Action:
    print("Parse segment: Checking token: " + tokens[0])
    for action in ALL_ACTIONS:
        if tokens[0] == action.START_CHAR:
            return action.parse_segment(tokens, ALL_START_CHARS, ALL_END_CHARS, parse_segment, tokenizer)
    # If we get here, it's a text segment
    return PromptSegment(tokens.pop(0), tokenizer)

def parse(tokens: list[str], tokenizer: SD1Tokenizer) -> list[PromptSegment | Action]:
    parsed = []
    while tokens:
        print("Parse: Checking token: " + tokens[0])
        if tokens[0] in ALL_START_CHARS:
            parsed.append(parse_segment(tokens, tokenizer))
        else:
            parsed.append(PromptSegment(tokens.pop(0), tokenizer))
    return parsed


def parse_special_tokens(string) -> list[str]:
    out = []
    current = ""

    for char in string:
        if char in ALL_START_CHARS:
            out += [current]
            current = char
        elif char in ALL_END_CHARS:
            out += [current + char]
            current = ""
        else:
            current += char
    out += [current]
    return out


def parse_segment_actions(string, tokenizer: SD1Tokenizer) -> list[PromptSegment | NudgeAction | ArithAction]:
    tokens = tokenize(string)
    parsed = parse(tokens, tokenizer)
    return parsed

class TokenDict:
    def __init__(self,
                 token_id: int,
                 weight: float = None,
                 nudge_id=None, nudge_weight=None, nudge_start: int = None, nudge_end: int = None,
                 arith_ops: dict[str, list[str]] = None):
        if weight is None:
            self.weight = 1.0
        else:
            self.weight = weight

        self.token_id = token_id
        self.nudge_id = nudge_id
        self.nudge_weight = nudge_weight
        self.nudge_index_start = nudge_start
        self.nudge_index_stop = nudge_end

        self.arith_ops = arith_ops


class MyTokenizer(SD1Tokenizer):
    def __init__(self, tokenizer_path=None, max_length=77, pad_with_end=True, embedding_directory=None, embedding_size=768, embedding_key='clip_l', special_tokens=None):
        super().__init__(tokenizer_path, max_length, pad_with_end, embedding_directory, embedding_size, embedding_key)

    """
    :return: list of tuples (tokenDict, word_id?)
    """
    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs) -> list[list[Action | int]]:
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0

        parsed_actions = parse_segment_actions(text, self)

        # nudge_start = kwargs.get("nudge_start")
        # nudge_end = kwargs.get("nudge_end")
        #
        # if nudge_start is not None and nudge_end is not None:
        #     nudge_start = int(nudge_start)
        #     nudge_end = int(nudge_end)
        #
        # # tokenize words
        for segment in parsed_actions:
            if isinstance(segment, Action):
                print(segment.depth_repr())
            else:
                print(segment.depth_repr())

        tokens: list[list[Action | int ]] = []

        # reshape token array to CLIP input size
        batched_tokens = []
        batch = [(TokenDict(token_id=self.start_token), 0)]
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            # determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - 1:
                    remaining_length = self.max_length - len(batch) - 1
                    # break word in two and add end token
                    if is_large:
                        batch.extend([(tokenDict, i+1) for tokenDict in t_group[:remaining_length]])
                        batch.append((TokenDict(token_id=self.end_token), 0))
                        t_group = t_group[remaining_length:]
                    # add end token and pad
                    else:
                        batch.append((TokenDict(token_id=self.end_token), 0))
                        batch.extend([(TokenDict(token_id=pad_token), 0)] * remaining_length)
                    # start new batch
                    batch = [(TokenDict(token_id=self.start_token), 1.0, 0)]
                    batched_tokens.append(batch)
                else:
                    batch.extend([(tokenDict, i+1) for tokenDict in t_group])
                    t_group = []

        # fill last batch
        batch.extend([(TokenDict(token_id=self.end_token), 0)] + [
            (TokenDict(token_id=pad_token), 0)] * (self.max_length - len(batch) - 1))

        if not return_word_ids:
            batched_tokens = [
                [
                    (tokenInfo[0],) for tokenInfo in batch
                ] for batch in batched_tokens
            ]

        return batched_tokens
