import torch
from torch.nn import Embedding
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextTransformer

from custom_nodes.KepPromptLang.lib.action.base import Action, SingleArgAction
from custom_nodes.KepPromptLang.lib.actions.action_utils import get_total_length
from custom_nodes.KepPromptLang.lib.fun_clip_stuff import (
    PromptLangCLIPTextEmbeddings,
    PrompLangCLIPTextTransformer,
)
from custom_nodes.KepPromptLang.lib.parser.prompt_segment import PromptSegment


class PoolerAction(SingleArgAction):
    grammar = 'pooler(" arg+ ")"'
    name = "_exp-pooler"
    chars = ["[", "]"]

    def __init__(self, args):
        super().__init__(args)
        self.result = None

    def token_length(self) -> int:
        """
        Pooler returns the embedding of the EOT token, so the length is 1
        """
        return 1

    def process_with_transformer(
        self, transformer: CLIPTextTransformer, embedding_module: Embedding
    ) -> None:
        """ """
        # SOT + tokens + EOT
        eot_token = embedding_module.num_embeddings - 1
        print("Using EOT token", eot_token)

        arg_length = get_total_length(self.arg)

        # SOT + arg length + EOT
        empty_tokens = [[49406] + [eot_token] * (arg_length + 1)]

        transformer_results: BaseModelOutputWithPooling = transformer(
            [
                [PromptSegment(text="_Empty Batch_", tokens=empty_tokens[0])],
                [PromptSegment(text="[SOT]", tokens=[49406])]
                + self.arg
                + [PromptSegment(text="[EOT]", tokens=[eot_token])],
            ]
        )
        self.result = transformer_results.pooler_output[1].unsqueeze(0).unsqueeze(0)

    def get_result(self, embedding_module: Embedding) -> torch.Tensor:
        if self.result is not None:
            return self.result

        raise Exception(
            "Pooled action result is not set. Did you forget to call process_with_transformer?"
        )
