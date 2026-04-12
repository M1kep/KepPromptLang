"""Verify the tokenizer emits ComfyUI's native (token, weight) format with lazy Actions.

Uses the comfy stub from conftest, so no real ComfyUI needed.
"""

import pytest

torch = pytest.importorskip("torch")

from KepPromptLang.lib.actions.base import Action
from KepPromptLang.lib.actions.sum import SumAction
from KepPromptLang.lib.tokenizer import PromptLangSDTokenizer


@pytest.fixture
def tok():
    return PromptLangSDTokenizer()


def _logical_length(row):
    """Post-splice length: ints/tensors count 1, Actions count token_length()."""
    total = 0
    for token, _weight in row:
        if isinstance(token, Action):
            total += token.token_length()
        else:
            total += 1
    return total


def test_plain_text_is_int_tuples(tok):
    [row] = tok.tokenize_with_weights("hello world")
    assert all(isinstance(t, int) and w == 1.0 for t, w in row)
    assert row[0] == (tok.start_token, 1.0)
    assert row[-1][0] in (tok.end_token, 0)
    assert _logical_length(row) == tok.max_length


def test_action_is_lazy_single_entry(tok):
    [row] = tok.tokenize_with_weights("a sum(king|man|woman) here")
    actions = [t for t, _ in row if isinstance(t, Action)]
    assert len(actions) == 1
    assert isinstance(actions[0], SumAction)
    # Action occupies one row entry but its token_length slots are reserved in padding.
    assert _logical_length(row) == tok.max_length
    assert len(row) == tok.max_length - (actions[0].token_length() - 1)


def test_nested_actions_stay_nested(tok):
    [row] = tok.tokenize_with_weights("sum(diff(king|man)|woman)")
    actions = [t for t, _ in row if isinstance(t, Action)]
    assert len(actions) == 1
    assert isinstance(actions[0], SumAction)
    # Nested DiffAction lives inside the SumAction's args, not as a separate row entry.
    from KepPromptLang.lib.actions.diff import DiffAction
    assert isinstance(actions[0].all_args[0][0], DiffAction)


def test_overflow_splits_into_multiple_batches(tok):
    # 80 single-token words won't fit in one 77-slot batch.
    text = " ".join(f"w{i}" for i in range(80))
    batches = tok.tokenize_with_weights(text)
    assert len(batches) >= 2
    for row in batches:
        assert _logical_length(row) == tok.max_length
        assert row[0] == (tok.start_token, 1.0)
