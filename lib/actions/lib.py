from custom_nodes.ClipStuff.lib.actions import ALL_START_CHARS, ALL_END_CHARS
from custom_nodes.ClipStuff.lib.actions.base import Action


def is_action_segment(action_class: Action.__class__, segment: str):
    if not issubclass(action_class, Action):
        raise Exception(
            f"action_class must be a subclass of Action, got {action_class}"
        )

    return (
        segment[0] == action_class.START_CHAR and segment[-1] == action_class.END_CHAR
    )


def is_any_action_segment(segment: str):
    return segment[0] in ALL_START_CHARS and segment[-1] in ALL_END_CHARS
