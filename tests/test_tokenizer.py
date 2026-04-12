"""Verify the tokenizer emits ComfyUI's native (token, weight) format with lazy Actions
and per-position weights.

Uses the comfy stub from conftest, so no real ComfyUI needed.
"""

import pytest

torch = pytest.importorskip("torch")

from KepPromptLang.lib.actions.base import ACTION_CONTINUATION, Action
from KepPromptLang.lib.actions.sum import SumAction
from KepPromptLang.lib.actions.weighted import WeightedGroup
from KepPromptLang.lib.tokenizer import PromptLangSDTokenizer


@pytest.fixture
def tok():
    return PromptLangSDTokenizer()


def test_plain_text_is_int_tuples_at_max_length(tok):
    [row] = tok.tokenize_with_weights("hello world")
    assert all(isinstance(t, int) and w == 1.0 for t, w in row)
    assert row[0] == (tok.start_token, 1.0)
    assert len(row) == tok.max_length


def test_action_emits_one_entry_plus_continuations(tok):
    [row] = tok.tokenize_with_weights("a sum(king|man|woman) here")
    assert len(row) == tok.max_length

    actions = [t for t, _ in row if isinstance(t, Action)]
    continuations = [t for t, _ in row if t is ACTION_CONTINUATION]
    assert len(actions) == 1
    assert isinstance(actions[0], SumAction)
    assert len(continuations) == actions[0].token_length() - 1


def test_paren_weight_syntax(tok):
    [row] = tok.tokenize_with_weights("a (cat:1.3) here")
    weighted = [(t, w) for t, w in row if w != 1.0]
    # "cat" is one token under the fake tokenizer.
    assert len(weighted) == 1
    assert weighted[0][1] == pytest.approx(1.3)
    assert isinstance(weighted[0][0], int)


def test_paren_weight_on_action_propagates_to_continuations(tok):
    [row] = tok.tokenize_with_weights("(sum(king|man|woman):0.7)")
    action_entry = next((t, w) for t, w in row if isinstance(t, Action))
    cont_weights = [w for t, w in row if t is ACTION_CONTINUATION]
    assert action_entry[1] == pytest.approx(0.7)
    assert all(w == pytest.approx(0.7) for w in cont_weights)


def test_nested_paren_weights_multiply(tok):
    [row] = tok.tokenize_with_weights("((cat:1.2):0.5)")
    weighted = [(t, w) for t, w in row if w != 1.0]
    assert len(weighted) == 1
    assert weighted[0][1] == pytest.approx(0.6)


def test_emph_is_alias_for_paren_weight(tok):
    [row] = tok.tokenize_with_weights("emph(cat|1.3)")
    weighted = [(t, w) for t, w in row if w != 1.0]
    assert len(weighted) == 1
    assert weighted[0][1] == pytest.approx(1.3)


def test_nested_actions_stay_nested(tok):
    [row] = tok.tokenize_with_weights("sum(diff(king|man)|woman)")
    actions = [t for t, _ in row if isinstance(t, Action)]
    assert len(actions) == 1
    assert isinstance(actions[0], SumAction)
    from KepPromptLang.lib.actions.diff import DiffAction
    assert isinstance(actions[0].all_args[0][0], DiffAction)


def test_overflow_splits_into_multiple_batches(tok):
    text = " ".join(f"w{i}" for i in range(80))
    batches = tok.tokenize_with_weights(text)
    assert len(batches) >= 2
    for row in batches:
        assert len(row) == tok.max_length
        assert row[0] == (tok.start_token, 1.0)


def test_weighted_group_token_length():
    from KepPromptLang.lib.parser.prompt_segment import PromptSegment

    grp = WeightedGroup([PromptSegment("a", [1, 2]), PromptSegment("b", [3])], 1.5)
    assert grp.token_length() == 3
