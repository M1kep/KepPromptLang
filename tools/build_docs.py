import importlib
import inspect
import os
from typing import List, Type

from custom_nodes.KepPromptLang.lib.action.base import Action


EXCLUDED_MODULES = ["utils.py", "action_utils.py", "types.py"]
def import_module_from_path(path: str):
    module_name = path.replace("/", ".")[:-3]
    return importlib.import_module(module_name, package="custom_nodes.KepPromptLang.lib.actions")


# Function to find and import action classes
def find_action_classes(directory: str) -> List[Type[Action]]:
    action_classes = []
    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("__") and filename not in EXCLUDED_MODULES:
            module_path = os.path.join(directory, filename)
            module = import_module_from_path(module_path)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Action) and obj is not Action and obj.__name__ != "MultiArgAction" and obj.__name__ != "SingleArgAction":
                    action_classes.append(obj)
    return action_classes


# Function to extract info from an action class
def extract_class_info(cls: Type[Action]) -> dict:
    class_info = {
        'class_name': cls.__name__,
        'properties': {
            'display_name': getattr(cls, 'display_name', None),
            'action_name': getattr(cls, 'action_name', None),
            'description': getattr(cls, 'description', None),
            'usage_examples': getattr(cls, 'usage_examples', None)
        }
    }
    return class_info

def escape_pipes(text: str) -> str:
    return text.replace('|', '\\|')

def generate_markdown_documentation(classes_info: List[dict]) -> str:
    documentation = "# Actions Documentation\n\n"

    # Define table columns
    # columns = ["Class", "Display Name", "Action Name", "Description", "Usage Examples"]
    columns = ["Display Name", "Action Name", "Description", "Usage Examples"]
    documentation += "| " + " | ".join(columns) + " |\n"
    documentation += "| --- " * len(columns) + "|\n"

    for cls_info in classes_info:
        # row = [cls_info['class_name']]
        row = []
        # Iterate over properties in a predefined order
        for prop in ["display_name", "action_name", "description", "usage_examples"]:
            prop_doc = cls_info['properties'].get(prop, 'N/A')

            # Format and escape usage examples
            if isinstance(prop_doc, list):
                escaped_examples = [escape_pipes(example) for example in prop_doc]
                prop_doc = "<ul>" + "".join([f"<li>{example}</li>" for example in escaped_examples]) + "</ul>"
            else:
                prop_doc = escape_pipes(prop_doc)

            row.append(prop_doc)

        documentation += "| " + " | ".join(row) + " |\n"

    return documentation



# Main execution
if __name__ == "__main__":
    actions_directory = (
        "../lib/actions"  # Update this path as per your project structure
    )
    action_classes = find_action_classes(actions_directory)
    class_infos = [extract_class_info(cls) for cls in action_classes]
    docs = generate_markdown_documentation(class_infos)
    print(docs)  # Or write to a file
