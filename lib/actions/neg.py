import torch
from torch.nn import Embedding

from custom_nodes.KepPromptLang.lib.action.base import Action, SingleArgAction
from custom_nodes.KepPromptLang.lib.parser.registration import register_action


class NegAction(SingleArgAction):
    grammar = 'neg(" arg+ ")"'
    name = "neg"
    chars = ["[", "]"]

    def token_length(self) -> int:
        """
        Neg negates the embeddings of the base segment, so the length is the length of the base segment
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
                all_embeddings.append(seg_or_action.get_result(embedding_module))
            else:
                all_embeddings.append(seg_or_action.get_embeddings(embedding_module))

        target_embeddings = torch.cat(all_embeddings, dim=1)
        return target_embeddings * -1

