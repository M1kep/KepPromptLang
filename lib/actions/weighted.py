from typing import List


class WeightedGroup:
    """A group of segments/actions sharing an attention weight (the `(text:1.2)` syntax)."""

    def __init__(self, items: List, weight: float):
        self.items = items
        self.weight = weight

    def token_length(self) -> int:
        return sum(item.token_length() for item in self.items)

    def __repr__(self) -> str:
        return f"({self.items}:{self.weight})"
