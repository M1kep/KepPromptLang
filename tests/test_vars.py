import pytest

torch = pytest.importorskip("torch")

from KepPromptLang.lib.actions.base import Action
from KepPromptLang.lib.actions.sum import SumAction
from KepPromptLang.lib.tokenizer import PromptLangSDTokenizer


@pytest.fixture
def tok():
    return PromptLangSDTokenizer()


def content_tokens(row, tok):
    """Non-SOT/EOT/pad int tokens from a row, in order."""
    return [
        t for t, _ in row
        if isinstance(t, int) and t not in (tok.start_token, tok.end_token, 0)
    ]


def test_var_substitutes_at_top_level(tok):
    [direct] = tok.tokenize_with_weights("a cat dog")
    [via_var] = tok.tokenize_with_weights("$x = cat dog; a $x")
    assert content_tokens(via_var, tok) == content_tokens(direct, tok)


def test_var_holding_action(tok):
    [row] = tok.tokenize_with_weights("$axis = sum(king|man); $axis")
    actions = [t for t, _ in row if isinstance(t, Action)]
    assert len(actions) == 1
    assert isinstance(actions[0], SumAction)


def test_var_inside_function_arg(tok):
    [row] = tok.tokenize_with_weights("$a = king; sum($a|woman)")
    actions = [t for t, _ in row if isinstance(t, Action)]
    assert len(actions) == 1
    # token_length should be 1 (single-token base arg via the fake tokenizer)
    assert actions[0].token_length() == 1


def test_var_under_weight(tok):
    [row] = tok.tokenize_with_weights("$x = cat; ($x:1.5)")
    weighted = [w for t, w in row if isinstance(t, int) and w != 1.0]
    assert weighted == [pytest.approx(1.5)]


def test_var_ref_before_assign_errors(tok):
    from lark.exceptions import VisitError
    with pytest.raises((ValueError, VisitError), match="referenced before assignment"):
        tok.tokenize_with_weights("$x and then $x = cat;")


def test_var_reassignment_errors(tok):
    from lark.exceptions import VisitError
    with pytest.raises((ValueError, VisitError), match="already defined"):
        tok.tokenize_with_weights("$x = cat; $x = dog; $x")


def test_var_chains(tok):
    [direct] = tok.tokenize_with_weights("cat")
    [chained] = tok.tokenize_with_weights("$a = cat; $b = $a; $b")
    assert content_tokens(chained, tok) == content_tokens(direct, tok)


def test_comments_ignored(tok):
    [a] = tok.tokenize_with_weights("cat dog")
    [b] = tok.tokenize_with_weights("cat # this is ignored\ndog")
    assert content_tokens(a, tok) == content_tokens(b, tok)


def test_assign_only_produces_empty_prompt(tok):
    [row] = tok.tokenize_with_weights("$x = cat;")
    # SOT + EOT + padding only
    assert content_tokens(row, tok) == []
