from ..parser.registration import register_action
from .avg import AverageAction
from .diff import DiffAction
from .mult import MultiplyAction
from .neg import NegAction
from .norm import NormAction
from .pos_scale import PosScaleAction
from .post_pos import PostPosAction
from .rand import RandAction
from .scale_dims import ScaleDims
from .set_dims import SetDims
from .slerp import SlerpAction
from .sum import SumAction

for _action in [
    AverageAction,
    DiffAction,
    MultiplyAction,
    NegAction,
    NormAction,
    PosScaleAction,
    PostPosAction,
    RandAction,
    ScaleDims,
    SetDims,
    SlerpAction,
    SumAction,
]:
    register_action(_action)
