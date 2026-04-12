"""Regenerate the action table in README.md.

Loads each action file by path so the docs can be regenerated without ComfyUI installed.
"""

import importlib
import inspect
import os
import sys
import types
from typing import List, Type

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIONS_DIR = os.path.join(REPO_ROOT, "lib", "actions")
EXCLUDED = {"__init__.py", "base.py", "types.py", "action_utils.py", "utils.py"}


def _stub_runtime_deps():
    """Stub modules whose only purpose is to satisfy the package's top-level imports."""
    sys.path.insert(0, os.path.dirname(REPO_ROOT))

    # Avoid pulling in ComfyUI's nodes.py during action discovery.
    pkg_init = sys.modules.get("KepPromptLang")
    if pkg_init is None:
        pkg = types.ModuleType("KepPromptLang")
        pkg.__path__ = [REPO_ROOT]
        sys.modules["KepPromptLang"] = pkg


def find_action_classes() -> List[Type]:
    _stub_runtime_deps()
    base_class = importlib.import_module("KepPromptLang.lib.actions.base").Action

    found: List[Type] = []
    for filename in sorted(os.listdir(ACTIONS_DIR)):
        if not filename.endswith(".py") or filename in EXCLUDED:
            continue
        mod = importlib.import_module(f"KepPromptLang.lib.actions.{filename[:-3]}")
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(cls, base_class)
                and cls is not base_class
                and cls.__module__ == mod.__name__
            ):
                found.append(cls)
    return found


def render_table(classes: List[Type]) -> str:
    rows = []
    for cls in sorted(classes, key=lambda c: c.action_name):
        examples = "<ul>" + "".join(
            f"<li>{ex.replace('|', chr(92) + '|')}</li>"
            for ex in (cls.usage_examples or [])
        ) + "</ul>"
        cells = [
            (cls.display_name or "").replace("|", "\\|"),
            (cls.action_name or "").replace("|", "\\|"),
            (cls.description or "").replace("|", "\\|"),
            examples,
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return (
        "| Display Name | Action Name | Description | Usage Examples |\n"
        "| --- | --- | --- | --- |\n"
        + "\n".join(rows)
        + "\n"
    )


if __name__ == "__main__":
    print(render_table(find_action_classes()))
