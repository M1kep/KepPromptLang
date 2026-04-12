from typing import List, Tuple

from torch import Tensor
from torch.nn import Embedding

from .action_utils import concat_embeddings, get_total_length, parse_numeric_arg
from .base import MultiArgAction, PostModifiers
from .types import SegOrAction


class PosScaleAction(MultiArgAction):
    grammar = 'posScale(" arg+ ")"'

    display_name = "Positional Embedding Scale"
    action_name = "posScale"
    description = (
        "Scales (multiplies) the positional embeddings of the provided segments or actions by the multiplier."
    )
    usage_examples = [
        "A posScale(cat|1.5) on a rainy day",
    ]

    def __init__(self, args: List[List[SegOrAction]]) -> None:
        super().__init__(args)
        if len(args) != 2:
            raise ValueError("PosScale action expects exactly two arguments")
        self.target_arg = args[0]
        self.parsed_multiplier = parse_numeric_arg(
            args[1], action_name="PosScale", role="multiplier", cast=float
        )

    def token_length(self) -> int:
        return get_total_length(self.target_arg)

    def get_result(self, embedding_module: Embedding) -> Tuple[Tensor, PostModifiers]:
        target_embeddings = concat_embeddings(self.target_arg, embedding_module)
        return target_embeddings, PostModifiers(position_embed_scale=self.parsed_multiplier)
