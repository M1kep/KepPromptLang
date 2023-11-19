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


class PooledAvgAction(SingleArgAction):
    grammar = 'pooledAvg(" arg+ ")"'
    chars = ["[", "]"]

    display_name = "Pooled Average(Experimental)"
    action_name = "_exp-pooledAvg"
    description = "Processes the provided segments or actions fully through CLIP and creates a pooled average of the last hidden state by averaging the last hidden state of each token."
    usage_examples = [
        "A cat on a _exp-pooledAvg(beautiful sunny day)",
        "A _exp-pooledAvg(broken glass) bottle",
    ]

    def __init__(self, args):
        super().__init__(args)
        self.result = None

    def token_length(self) -> int:
        """
        PooledAvg returns the average of the last hidden state, so the length is 1
        """
        return 1

    def process_with_transformer(
        self, transformer: CLIPTextTransformer, embedding_module: Embedding
    ) -> None:
        """ """
        # SOT + tokens + EOT
        eot_token = embedding_module.num_embeddings - 1
        print("Using EOT token", eot_token)
        #TODO: Play with impact of padding on pooled output
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
        self.result = (
            transformer_results.last_hidden_state[1, 1:-1, :].mean(dim=0).unsqueeze(0).unsqueeze(0)
        )

    def get_result(self, embedding_module: Embedding) -> torch.Tensor:
        if self.result is not None:
            return self.result

        raise Exception(
            "PooledAvg action result is not set. Did you forget to call process_with_transformer?"
        )
