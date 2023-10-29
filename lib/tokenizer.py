from typing import List, Dict

from lark import Tree

from comfy.sd1_clip import SD1Tokenizer, SDTokenizer
from custom_nodes.KepPromptLang.lib.actions.types import SegOrAction

from custom_nodes.KepPromptLang.lib.parser import PromptParser
from custom_nodes.KepPromptLang.lib.parser.transformer import PromptTransformer
from custom_nodes.KepPromptLang.lib.parser.prompt_segment import PromptSegment

class PromptLangTokenizer(SD1Tokenizer):
    def __init__(self, embedding_directory=None, clip_name='l', tokenizer=SDTokenizer) -> None:
        super().__init__(embedding_directory, clip_name, tokenizer)

    """
    Doesn't actually tokenize...
    Returns batches of segments and actions
    :return: List of list(batches) of segments and actions
    """
    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs) -> List[List[SegOrAction]]:
        clip: SDTokenizer = getattr(self, self.clip)
        if clip.pad_with_end:
            pad_token = clip.end_token
        else:
            pad_token = 0

        parsed_prompt = PromptParser.parse(text)
        parsed_actions = PromptTransformer(self).transform(parsed_prompt)

        # reshape token array to CLIP input size
        batched_segments = []
        batch = [PromptSegment(text="[SOT]", tokens=[clip.start_token])]
        # batched_segments.append(batch)
        batch_size = 1
        if isinstance(parsed_actions, Tree):
            segments_to_process = parsed_actions.children
        else:
            segments_to_process = [parsed_actions]
        for segment in segments_to_process:
            num_tokens = segment.token_length()
            # determine if we're going to try and keep the tokens in a single batch
            is_large = num_tokens >= clip.max_word_length

            # If the segment is too large to fit in a single batch, pad the current batch and start a new one
            if num_tokens + batch_size > clip.max_length - 1:
                remaining_length = clip.max_length - batch_size
                # Pad batch
                batch.append(PromptSegment("__PAD__", [clip.end_token] + [pad_token] * (remaining_length - 1))) # -1 for end token
                batched_segments.append(batch)

                # start new batch
                batch = [PromptSegment(text="[SOT]", tokens=[clip.start_token]), segment]
                batch_size = num_tokens + 1 # +1 for start token
                continue

            # Since the segment fits in the current batch, add it
            batch.append(segment)
            batch_size += num_tokens

        # Pad the last batch
        remaining_length = clip.max_length - batch_size - 1 # -1 for end token
        batch.append(PromptSegment("__PAD__", [clip.end_token] + [pad_token] * remaining_length))
        batched_segments.append(batch)

        # for batch in batched_segments:
        #     batch_size_info(batch)

        return batched_segments


class PromptLangSDXLClipGTokenizer(PromptLangTokenizer):
    def __init__(self, tokenizer_path=None, embedding_directory=None) -> None:
        super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=1280, embedding_key='clip_g')


class PromptLangSDXLTokenizer(SD1Tokenizer):
    def __init__(self, embedding_directory=None) -> None:
        self.clip_l = PromptLangTokenizer(embedding_directory=embedding_directory)
        self.clip_g = PromptLangSDXLClipGTokenizer(embedding_directory=embedding_directory)

    def tokenize_with_weights(self, text:str, return_word_ids=False) -> Dict[str, List[List[SegOrAction]]]:
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text, return_word_ids)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)
        return out
