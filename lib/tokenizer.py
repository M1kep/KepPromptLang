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
from custom_nodes.ClipStuff.lib.actions.base import Action
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



def parse_segment(tokens: list[str]) -> Union[str, Action]:
    for action in ALL_ACTIONS:
        if tokens[0] == action.START_CHAR:
            return action.parse_segment(tokens, ALL_START_CHARS, ALL_END_CHARS, parse_segment)
    # If we get here, it's a word
    return tokens.pop(0)
def parse(tokens: list[str]) -> list[Union[str, Action]]:
    parsed = []
    while tokens:
        if tokens[0] in ALL_START_CHARS:
            parsed.append(parse_segment(tokens))
        else:
            parsed.append(tokens.pop(0))
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


def parse_segment_actions(string) -> list[Union[str, NudgeAction, ArithAction]]:
    tokens = tokenize(string)
    parsed = parse(tokens)
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

        parsed_actions = parse_segment_actions(text)

        nudge_start = kwargs.get("nudge_start")
        nudge_end = kwargs.get("nudge_end")

        if nudge_start is not None and nudge_end is not None:
            nudge_start = int(nudge_start)
            nudge_end = int(nudge_end)

        # tokenize words
        tokens: list[list[Action | int ]] = []

        for action in parsed_actions:
            nudge_weight = None
            nudge_to_id = None
            arith_ops = None
            if isinstance(action, str):
                segment_to_tokenize = action
            elif isinstance(action, NudgeAction):
                segment_to_tokenize = action.base_segment
                nudge_to_id = self.tokenizer(action.target)["input_ids"][1:-1][0]

                nudge_weight = action.weight
                if nudge_weight is None:
                    nudge_weight = 0.5
            elif isinstance(action, ArithAction):
                segment_to_tokenize = action.base_segment
                arith_ops = action.ops
                for op in arith_ops:
                    arith_ops[op] = [self.tokenizer(word)["input_ids"][1:-1][0] for word in arith_ops[op]]
            else:
                raise Exception(f"Unexpected action type: {type(action)}")

            to_tokenize = segment_to_tokenize.split(' ')
            to_tokenize = [x for x in to_tokenize if x != ""]

            for word in to_tokenize:
                # if we find an embedding, deal with the embedding
                if word.startswith(self.embedding_identifier) and self.embedding_directory is not None:
                    embedding_name = word[len(self.embedding_identifier):].strip('\n')
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        print(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([TokenDict(token_id=embed)])
                        else:
                            tokens.append([
                                TokenDict(token_id=embed[x])
                                for x in range(embed.shape[0])
                            ])
                    # if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                # parse word
                tokens.append([t] for t in self.tokenizer(word)["input_ids"][1:-1])
                tokens.append([TokenDict(
                    token_id=t,
                    nudge_id=nudge_to_id,
                    nudge_weight=nudge_weight,
                    nudge_start=nudge_start,
                    nudge_end=nudge_end,
                    arith_ops=arith_ops
                ) for t in self.tokenizer(word)["input_ids"][1:-1]])

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
