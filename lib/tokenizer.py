from typing import Union, TypedDict, Optional

from comfy.sd1_clip import SD1Tokenizer, escape_important, unescape_important, parse_parentheses


def token_weights(string, current_weight):
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ')' and x[0] == '(':
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx+1:])
                    x = x[:xx]
                except:
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out

def parse_brackets(string):
    out = []
    current = ""
    for char in string:
        if char == '[':
            out += [current]
            current = "["
        elif char == ']':
            out += [current + ']']
            current = ""
        else:
            current += char
    out += [current]
    return out

def parse_nudges(string) -> list[tuple[str, Union[str, None]]]:
    out = []
    for nudge_segment in parse_brackets(string):
        if nudge_segment == "":
            continue

        if nudge_segment[0] != '[' and nudge_segment[-1] != ']':
            out += [(nudge_segment, None, None)]
            continue

        nudge_segment = nudge_segment[1:-1]
        sep_idx = nudge_segment.find(":")
        if sep_idx < 0:
            out += [(nudge_segment, None, None)]
            continue

        nudge_to = nudge_segment[sep_idx+1:]
        weight = None

        weight_sep_idx = nudge_to.find(":")
        if weight_sep_idx >= 0:
            [nudge_to, weight] = nudge_to.split(":")
            weight = float(weight)

        out += [(nudge_segment[:sep_idx], nudge_to, weight)]
    return out

# class TokenDict(TypedDict):
#     token_id: int
#     weight: float
#     nudge_id: Optional[int]
#     nudge_weight: Optional[float]
#

class TokenDict:
    def __init__(self, token_id: int, weight: float = None, nudge_id=None, nudge_weight=None, nudge_start: int = None,
                 nudge_end: int = None):
        if weight is None:
            self.weight = 1.0
        else:
            self.weight = weight

        self.token_id = token_id
        self.nudge_id = nudge_id
        self.nudge_weight = nudge_weight
        self.nudge_index_start = nudge_start
        self.nudge_index_stop = nudge_end



class MyTokenizer(SD1Tokenizer):
    def __init__(self, tokenizer_path=None, max_length=77, pad_with_end=True, embedding_directory=None, embedding_size=768, embedding_key='clip_l', special_tokens=None):
        super().__init__(tokenizer_path, max_length, pad_with_end, embedding_directory, embedding_size, embedding_key)

    """
    :return: list of tuples (tokenDict, word_id?)
    """
    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0

        parsed_nudges = parse_nudges(text)

        nudge_start = None
        nudge_end = None
        if kwargs.get('nudge_start', None) is not None and kwargs.get('nudge_end', None) is not None:
            nudge_start = int(kwargs.get('nudge_start'))
            nudge_end = int(kwargs.get('nudge_end'))


        #tokenize words
        tokens: list[list[TokenDict]] = []

        for token_segment, nudge_to_token, nudge_weight in parsed_nudges:
            to_tokenize = token_segment.split(' ')
            to_tokenize = [x for x in to_tokenize if x != ""]
            # if token_segment == ' ':
            #     continue

            if nudge_weight is None:
                nudge_weight = .5

            nudge_to_id = None
            if nudge_to_token is not None:
                # self.convert_tokens_to_ids
                nudge_to_id = self.tokenizer(nudge_to_token)["input_ids"][1:-1][0]

            for word in to_tokenize:
                #if we find an embedding, deal with the embedding
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
                    #if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                #parse word
                tokens.append([TokenDict(
                    token_id=t,
                    nudge_id=nudge_to_id,
                    nudge_weight=nudge_weight,
                    nudge_start=nudge_start,
                    nudge_end=nudge_end
                ) for t in self.tokenizer(word)["input_ids"][1:-1]])

        #reshape token array to CLIP input size
        batched_tokens = []
        batch = [(TokenDict(token_id=self.start_token), 0)]
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            #determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - 1:
                    remaining_length = self.max_length - len(batch) - 1
                    #break word in two and add end token
                    if is_large:
                        batch.extend([(tokenDict, i+1) for tokenDict in t_group[:remaining_length]])
                        batch.append((TokenDict(token_id=self.end_token), 0))
                        t_group = t_group[remaining_length:]
                    #add end token and pad
                    else:
                        batch.append((TokenDict(token_id=self.end_token), 0))
                        batch.extend([(TokenDict(token_id=pad_token), 0)] * (remaining_length))
                    #start new batch
                    batch = [(TokenDict(token_id=self.start_token), 1.0, 0)]
                    batched_tokens.append(batch)
                else:
                    batch.extend([(tokenDict,i+1) for tokenDict in t_group])
                    t_group = []

        #fill last batch
        batch.extend([(TokenDict(token_id=self.end_token), 0)] + [
            (TokenDict(token_id=pad_token), 0)] * (self.max_length - len(batch) - 1))

        if not return_word_ids:
            batched_tokens = [
                [
                    (tokenInfo[0],) for tokenInfo in batch
                ] for batch in batched_tokens
            ]

        return batched_tokens
