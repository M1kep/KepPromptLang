from comfy.sd1_clip import SD1Tokenizer
from custom_nodes.ClipStuff.lib.action.base import (
    Action,
)

from custom_nodes.ClipStuff.lib.parser import PromptParser
from custom_nodes.ClipStuff.lib.parser.transformer import PromptTransformer
from custom_nodes.ClipStuff.lib.parser.prompt_segment import PromptSegment

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
    Doesn't actually tokenize...
    Returns batches of segments and actions
    :return: List of list(batches) of segments and actions
    """
    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs) -> list[list[PromptSegment | Action]]:
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0

        parsed_prompt = PromptParser.parse(text)
        parsed_actions = PromptTransformer(self).transform(parsed_prompt)

        # reshape token array to CLIP input size
        batched_segments = []
        batch = [PromptSegment(text="[SOT]", tokens=[self.start_token])]
        # batched_segments.append(batch)
        batch_size = 1
        for segment in parsed_actions.children:
            num_tokens = segment.token_length()
            # determine if we're going to try and keep the tokens in a single batch
            is_large = num_tokens >= self.max_word_length

            # If the segment is too large to fit in a single batch, pad the current batch and start a new one
            if num_tokens + batch_size > self.max_length - 1:
                remaining_length = self.max_length - batch_size - 1 # -1 for end token
                # Pad batch
                batch.append(PromptSegment("__PAD__", [self.end_token] + [pad_token] * remaining_length - 1))
                batched_segments.append(batch)

                # start new batch
                batch = [PromptSegment(text="[SOT]", tokens=[self.start_token]), segment]
                batch_size = num_tokens + 1 # +1 for start token
                continue

            # If the segment is small enough to fit in the current batch, add it
            batch.append(segment)
            batch_size += num_tokens

        # Pad the last batch
        remaining_length = self.max_length - batch_size - 1 # -1 for end token
        batch.append(PromptSegment("__PAD__", [self.end_token] + [pad_token] * remaining_length))
        batched_segments.append(batch)

        # for batch in batched_segments:
        #     batch_size_info(batch)

        return batched_segments
