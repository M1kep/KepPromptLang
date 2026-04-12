from KepPromptLang.lib.actions.diff import DiffAction
from KepPromptLang.lib.actions.norm import NormAction
from KepPromptLang.lib.actions.sum import SumAction
from KepPromptLang.lib.parser import PromptParser
from KepPromptLang.lib.parser.prompt_segment import PromptSegment
from KepPromptLang.lib.parser.transformer import PromptTransformer


def parse(text, tokenizer):
    tree = PromptParser.parse(text)
    return PromptTransformer(tokenizer).transform(tree)


def test_plain_words_become_segments(tokenizer):
    result = parse("hello world", tokenizer)
    items = result.children
    assert len(items) == 2
    assert all(isinstance(i, PromptSegment) for i in items)
    assert items[0].text == "hello"
    assert items[1].text == "world"


def test_sum_action_parses(tokenizer):
    action = parse("sum(king|man|woman)", tokenizer)
    items = action.children if hasattr(action, "children") else [action]
    assert len(items) == 1
    assert isinstance(items[0], SumAction)
    assert len(items[0].all_args) == 3


def test_nested_actions(tokenizer):
    action = parse("sum(diff(king|man)|woman)", tokenizer)
    items = action.children if hasattr(action, "children") else [action]
    outer = items[0]
    assert isinstance(outer, SumAction)
    inner = outer.all_args[0][0]
    assert isinstance(inner, DiffAction)


def test_norm_single_arg(tokenizer):
    action = parse("norm(cat)", tokenizer)
    items = action.children if hasattr(action, "children") else [action]
    assert isinstance(items[0], NormAction)


def test_quoted_string(tokenizer):
    result = parse('"hello world"', tokenizer)
    items = result.children if hasattr(result, "children") else [result]
    assert isinstance(items[0], PromptSegment)
    assert items[0].text == "hello world"


def test_unknown_action_errors(tokenizer):
    import pytest
    from lark.exceptions import VisitError

    with pytest.raises((ValueError, VisitError), match="not found in registry"):
        parse("nonexistentAction(cat)", tokenizer)
