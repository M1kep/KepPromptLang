from .arith import ArithAction
from .nudge import NudgeAction

ALL_ACTIONS = [NudgeAction, ArithAction]
ALL_START_CHARS = [action.START_CHAR for action in ALL_ACTIONS]
ALL_END_CHARS = [action.END_CHAR for action in ALL_ACTIONS]
