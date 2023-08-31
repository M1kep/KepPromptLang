import torch
from torch.nn import Embedding

from custom_nodes.KepPromptLang.lib.action.base import (
    Action,
    SingleArgAction,
)


class RandAction(SingleArgAction):
    grammar = 'rand(" arg ")"'
    name = "rand"
    chars = None

    parsed_token_length = 0

    def __init__(self, arg):
        super().__init__(arg)

        if len(self.arg) != 1:
            raise ValueError("Random action should have exactly one argument")

        seg_or_action = self.arg[0]

        if isinstance(seg_or_action, Action):
            raise ValueError("Random action should not have an action as an argument")

        try:
            self.parsed_token_length = int(seg_or_action.text)
        except ValueError:
            raise ValueError("Random action should have an integer as an argument")

    def token_length(self) -> int:
        """
        Random returns a random embedding whose length is the number in the argument
        :return:
        """
        return self.parsed_token_length

    def get_result(self, embedding_module: Embedding) -> torch.Tensor:
        # Create random tensor of size
        result = torch.rand((1, self.parsed_token_length, embedding_module.embedding_dim))
        return result

