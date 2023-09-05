from typing import Type, Dict

from custom_nodes.KepPromptLang.lib.action.base import Action

action_registry: Dict[str, Type[Action]] = {}

def register_action(action: Type[Action]) -> None:
    """

    :rtype: object
    """
    if action.name in action_registry:
        raise ValueError(f"Action {action.name} already registered")
    action_registry[str(action.name)] = action

def get_action_by_name(name: str) -> Type[Action]:
    if name not in action_registry:
        raise ValueError(f"Action {name} not found in registry")

    return action_registry[name]
