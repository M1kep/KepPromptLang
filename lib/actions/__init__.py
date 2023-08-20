from custom_nodes.ClipStuff.lib.actions.arith import ArithAction
from custom_nodes.ClipStuff.lib.actions.nudge import NudgeAction

ALL_ACTIONS = [NudgeAction, ArithAction]
ALL_START_CHARS = [action.START_CHAR for action in ALL_ACTIONS]
ALL_END_CHARS = [action.END_CHAR for action in ALL_ACTIONS]
