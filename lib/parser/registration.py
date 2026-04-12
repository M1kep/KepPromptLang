from typing import Dict, Type

from ..actions.base import Action

action_registry: Dict[str, Type[Action]] = {}


def register_action(action: Type[Action]) -> None:
    name = str(action.action_name)
    if name in action_registry:
        raise ValueError(f"Action {name} already registered")
    action_registry[name] = action


def get_action_by_name(name: str) -> Type[Action]:
    if name not in action_registry:
        raise ValueError(f"Action {name} not found in registry")
    return action_registry[name]
