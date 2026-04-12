"""Action-level tests using a tiny in-memory torch.nn.Embedding.

These verify the math/shape contracts of each action without needing ComfyUI or a real CLIP.
"""

import pytest

torch = pytest.importorskip("torch")

from KepPromptLang.lib.actions.avg import AverageAction
from KepPromptLang.lib.actions.diff import DiffAction
from KepPromptLang.lib.actions.mult import MultiplyAction
from KepPromptLang.lib.actions.neg import NegAction
from KepPromptLang.lib.actions.norm import NormAction
from KepPromptLang.lib.actions.pos_scale import PosScaleAction
from KepPromptLang.lib.actions.post_pos import PostPosAction
from KepPromptLang.lib.actions.rand import RandAction
from KepPromptLang.lib.actions.scale_dims import ScaleDims
from KepPromptLang.lib.actions.set_dims import SetDims
from KepPromptLang.lib.actions.slerp import SlerpAction
from KepPromptLang.lib.actions.sum import SumAction
from KepPromptLang.lib.actions.utils import slerp
from KepPromptLang.lib.parser.prompt_segment import PromptSegment

EMBED_DIM = 4
VOCAB = 100


@pytest.fixture
def embedding():
    torch.manual_seed(0)
    emb = torch.nn.Embedding(VOCAB, EMBED_DIM)
    return emb


def seg(*token_ids):
    return PromptSegment(text="x", tokens=list(token_ids))


def test_sum_adds_embeddings(embedding):
    a = seg(1, 2)
    b = seg(3, 4)
    action = SumAction([[a], [b]])
    expected = embedding(torch.LongTensor([[1, 2]])) + embedding(torch.LongTensor([[3, 4]]))
    assert torch.allclose(action.get_result(embedding), expected)


def test_diff_subtracts_embeddings(embedding):
    a = seg(1, 2)
    b = seg(3, 4)
    action = DiffAction([[a], [b]])
    expected = embedding(torch.LongTensor([[1, 2]])) - embedding(torch.LongTensor([[3, 4]]))
    assert torch.allclose(action.get_result(embedding), expected)


def test_neg_negates(embedding):
    action = NegAction([seg(1, 2)])
    expected = -embedding(torch.LongTensor([[1, 2]]))
    assert torch.allclose(action.get_result(embedding), expected)


def test_mult_scales(embedding):
    # The multiplier is read from PromptSegment.text (mimicking parser output).
    target_seg = PromptSegment(text="3.5", tokens=[1])
    action = MultiplyAction([[seg(1, 2)], [target_seg]])
    expected = embedding(torch.LongTensor([[1, 2]])) * 3.5
    assert torch.allclose(action.get_result(embedding), expected)


def test_norm_unit_length(embedding):
    action = NormAction([seg(1, 2)])
    result = action.get_result(embedding)
    norms = torch.norm(result, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_avg_weighted_mix(embedding):
    weight_seg = PromptSegment(text="0.25", tokens=[1])
    action = AverageAction([[seg(1, 2)], [seg(3, 4)], [weight_seg]])
    expected = embedding(torch.LongTensor([[1, 2]])) * 0.75 + embedding(torch.LongTensor([[3, 4]])) * 0.25
    assert torch.allclose(action.get_result(embedding), expected)


def test_avg_mismatched_lengths_errors():
    weight_seg = PromptSegment(text="0.5", tokens=[1])
    with pytest.raises(ValueError, match="same length"):
        AverageAction([[seg(1, 2)], [seg(3)], [weight_seg]])


def test_slerp_endpoints(embedding):
    weight0 = PromptSegment(text="0.0", tokens=[1])
    weight1 = PromptSegment(text="1.0", tokens=[1])
    a, b = seg(1, 2), seg(3, 4)
    a_emb = embedding(torch.LongTensor([[1, 2]]))
    b_emb = embedding(torch.LongTensor([[3, 4]]))

    assert torch.allclose(SlerpAction([[a], [b], [weight0]]).get_result(embedding), a_emb, atol=1e-5)
    assert torch.allclose(SlerpAction([[a], [b], [weight1]]).get_result(embedding), b_emb, atol=1e-5)


def test_slerp_helper_endpoints_and_midpoint():
    low = torch.tensor([1.0, 0.0])
    high = torch.tensor([0.0, 1.0])  # 90 degrees apart, both unit length
    assert torch.allclose(slerp(0.0, low, high), low, atol=1e-5)
    assert torch.allclose(slerp(1.0, low, high), high, atol=1e-5)
    midpoint = slerp(0.5, low, high)
    # Midpoint of orthogonal unit vectors on the unit sphere is (sqrt(2)/2, sqrt(2)/2).
    expected = torch.tensor([2 ** 0.5 / 2, 2 ** 0.5 / 2])
    assert torch.allclose(midpoint, expected, atol=1e-5)


def test_rand_token_length_and_bounds():
    length_seg = PromptSegment(text="3", tokens=[1])
    min_seg = PromptSegment(text="-2", tokens=[1])
    max_seg = PromptSegment(text="2", tokens=[1])
    action = RandAction([[length_seg], [min_seg], [max_seg]])

    emb = torch.nn.Embedding(VOCAB, EMBED_DIM)
    result = action.get_result(emb)
    assert result.shape == (1, 3, EMBED_DIM)
    assert (result >= -2).all() and (result <= 2).all()


def test_scale_dims_modifies_only_target_dim(embedding):
    pair_seg = PromptSegment(text="0,3.0", tokens=[1])
    action = ScaleDims([[seg(1, 2)], [pair_seg]])
    base = embedding(torch.LongTensor([[1, 2]])).clone()
    result = action.get_result(embedding)
    assert torch.allclose(result[0, :, 0], base[0, :, 0] * 3.0)
    assert torch.allclose(result[0, :, 1:], base[0, :, 1:])


def test_set_dims_overwrites_value(embedding):
    pair_seg = PromptSegment(text="2,-9.5", tokens=[1])
    action = SetDims([[seg(1, 2)], [pair_seg]])
    result = action.get_result(embedding)
    assert torch.allclose(result[0, :, 2], torch.tensor([-9.5, -9.5]))


def test_pos_scale_returns_modifier(embedding):
    multiplier = PromptSegment(text="1.5", tokens=[1])
    action = PosScaleAction([[seg(1, 2)], [multiplier]])
    tensor, modifiers = action.get_result(embedding)
    assert tensor.shape == (1, 2, EMBED_DIM)
    assert modifiers.position_embed_scale == 1.5


def test_post_pos_returns_bypass(embedding):
    action = PostPosAction([seg(1, 2)])
    tensor, modifiers = action.get_result(embedding)
    assert tensor.shape == (1, 2, EMBED_DIM)
    assert modifiers.bypass_pos_embed is True


def test_action_token_lengths():
    a, b = seg(1, 2, 3), seg(4, 5, 6)
    assert SumAction([[a], [b]]).token_length() == 3
    assert DiffAction([[a], [b]]).token_length() == 3
    assert NegAction([a]).token_length() == 3
    assert NormAction([a]).token_length() == 3
