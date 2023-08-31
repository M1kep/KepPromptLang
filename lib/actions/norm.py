import torch
from torch.nn import Embedding

from custom_nodes.KepPromptLang.lib.action.base import (
    Action,
    SingleArgAction,
)


class NormAction(SingleArgAction):
    grammar = 'norm(" arg+ ")"'
    name = "norm"
    chars = None

    def token_length(self) -> int:
        """
        Norm normalizes the embeddings of the base segment, so the length is the length of the base segment
        :return:
        """
        total_length = 0
        for seg_or_action in self.arg:
            total_length += seg_or_action.token_length()

        return total_length

    def get_result(self, embedding_module: Embedding) -> torch.Tensor:
        all_embeddings = []
        for seg_or_action in self.arg:
            if isinstance(seg_or_action, Action):
                target_embeddings = seg_or_action.get_result(embedding_module)
            else:
                target_embeddings = seg_or_action.get_embeddings(embedding_module)
            all_embeddings.append(target_embeddings)

        target_embeddings = torch.cat(all_embeddings, dim=1)
        return torch.div(target_embeddings, torch.norm(target_embeddings, dim=-1, keepdim=True))

