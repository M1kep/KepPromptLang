from typing import List, Tuple

from torch import Tensor
from torch.nn import Embedding

from ..parser.prompt_segment import PromptSegment
from .action_utils import concat_embeddings, get_total_length
from .base import Action, MultiArgAction
from .types import SegOrAction


class ScaleDims(MultiArgAction):
    grammar = 'scaleDims(" arg ("|" arg)* ")"'

    display_name = "Scale Dimensions"
    action_name = "scaleDims"
    description = "Scales the specified dimensions of the input embeddings by the specified amount"
    usage_examples = [
        "The scaleDims(cat|4,1.5|76,1.2) is happy",
    ]

    def __init__(self, args: List[List[SegOrAction]]):
        super().__init__(args)
        self.base_arg = args[0]
        self.scale_args: List[Tuple[int, float]] = _parse_dim_value_pairs(args[1:], action_name="ScaleDims")

    def token_length(self) -> int:
        return get_total_length(self.base_arg)

    def get_result(self, embedding_module: Embedding) -> Tensor:
        embeddings = concat_embeddings(self.base_arg, embedding_module)
        for dim, scale in self.scale_args:
            embeddings[0, :, dim] *= scale
        return embeddings


def _parse_dim_value_pairs(
    args: List[List[SegOrAction]],
    *,
    action_name: str,
) -> List[Tuple[int, float]]:
    """Parse args of the form `<dim>,<value>` into `(int, float)` pairs.

    Used by both scaleDims and setDims.
    """
    pairs: List[Tuple[int, float]] = []
    for arg in args:
        if isinstance(arg, Action):
            raise ValueError(f"{action_name} args must be in the format <dim>,<value> but got an action")
        if len(arg) != 1:
            raise ValueError(f"{action_name} args must be a single segment of <dim>,<value>")

        seg = arg[0]
        assert isinstance(seg, PromptSegment)
        if "," not in seg.text:
            raise ValueError(f"{action_name} args must be <dim>,<value> but got: {seg.text!r}")

        dim_str, value_str = seg.text.split(",", 1)
        try:
            dim = int(dim_str)
        except ValueError:
            raise ValueError(f"{action_name} dim must be an integer; got {dim_str!r}")
        try:
            value = float(value_str)
        except ValueError:
            raise ValueError(f"{action_name} value must be a float; got {value_str!r}")
        pairs.append((dim, value))
    return pairs
