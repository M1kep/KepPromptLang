from ..parser.registration import register_action
from .avg import AverageAction
from .diff import DiffAction
from .mult import MultiplyAction
from .nearest import NearestAction
from .neg import NegAction
from .noise import NoiseAction
from .norm import NormAction
from .pos_scale import PosScaleAction
from .post_pos import PostPosAction
from .project import ProjectAction, RejectAction
from .rand import RandAction
from .renorm import RenormAction
from .scale_dims import ScaleDims
from .set_dims import SetDims
from .slerp import SlerpAction
from .sum import SumAction

for _action in [
    AverageAction,
    DiffAction,
    MultiplyAction,
    NearestAction,
    NegAction,
    NoiseAction,
    NormAction,
    PosScaleAction,
    PostPosAction,
    ProjectAction,
    RandAction,
    RejectAction,
    RenormAction,
    ScaleDims,
    SetDims,
    SlerpAction,
    SumAction,
]:
    register_action(_action)


class _LerpAlias(AverageAction):
    """`lerp(a|b|t)` is sugar for `avg(a|b|t)`."""

    display_name = "Lerp"
    action_name = "lerp"
    usage_examples = ["lerp(cat|dog|0.5)"]


register_action(_LerpAlias)
