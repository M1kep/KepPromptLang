from typing import Optional

from custom_nodes.ClipStuff.lib.actions.base import Action


class NudgeAction(Action):
    START_CHAR = "["
    END_CHAR = "]"

    def __init__(self, base_segment=None, weight: Optional[float] = None, target=None):
        self.base_segment = base_segment
        self.weight = weight
        self.target = target
