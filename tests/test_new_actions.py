import pytest

torch = pytest.importorskip("torch")

from KepPromptLang.lib.actions.nearest import NearestAction
from KepPromptLang.lib.actions.noise import NoiseAction
from KepPromptLang.lib.actions.project import ProjectAction, RejectAction
from KepPromptLang.lib.actions.renorm import RenormAction
from KepPromptLang.lib.parser.prompt_segment import PromptSegment
from KepPromptLang.lib.parser.registration import get_action_by_name

EMBED_DIM = 4
VOCAB = 50


@pytest.fixture
def embedding():
    torch.manual_seed(0)
    return torch.nn.Embedding(VOCAB, EMBED_DIM)


def seg(*token_ids):
    return PromptSegment(text="x", tokens=list(token_ids))


def test_proj_plus_reject_reconstructs_input(embedding):
    a, b = seg(1, 2), seg(3)
    proj = ProjectAction([[a], [b]]).get_result(embedding)
    rej = RejectAction([[a], [b]]).get_result(embedding)
    a_emb = embedding(torch.LongTensor([[1, 2]]))
    assert torch.allclose(proj + rej, a_emb, atol=1e-5)


def test_reject_is_orthogonal_to_b(embedding):
    a, b = seg(1, 2), seg(3)
    rej = RejectAction([[a], [b]]).get_result(embedding)
    b_dir = torch.nn.functional.normalize(
        embedding(torch.LongTensor([[3]])).mean(dim=1, keepdim=True), dim=-1
    )
    dots = (rej * b_dir).sum(dim=-1)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5)


def test_renorm_matches_ref_norm(embedding):
    a, ref = seg(1, 2), seg(3)
    out = RenormAction([[a], [ref]]).get_result(embedding)
    ref_norm = torch.norm(embedding(torch.LongTensor([[3]])), dim=-1).mean()
    out_norms = torch.norm(out, dim=-1)
    assert torch.allclose(out_norms, ref_norm.expand_as(out_norms), atol=1e-5)


def test_noise_shape_and_mean(embedding):
    std = PromptSegment(text="0.01", tokens=[1])
    out = NoiseAction([[seg(1, 2)], [std]]).get_result(embedding)
    base = embedding(torch.LongTensor([[1, 2]]))
    assert out.shape == base.shape
    # Perturbation magnitude bounded (5σ with margin); std=0.01, EMBED_DIM=4.
    assert (out - base).abs().max() < 0.2


def test_noise_zero_std_is_identity(embedding):
    std = PromptSegment(text="0.0", tokens=[1])
    out = NoiseAction([[seg(1, 2)], [std]]).get_result(embedding)
    base = embedding(torch.LongTensor([[1, 2]]))
    assert torch.allclose(out, base)


def test_nearest_returns_exact_token_for_that_token(embedding):
    out = NearestAction([[seg(7)]]).get_result(embedding)
    assert out.shape == (1, 1, EMBED_DIM)
    assert torch.allclose(out[0, 0], embedding.weight[7])


def test_nearest_k_tokens(embedding):
    k = PromptSegment(text="3", tokens=[1])
    action = NearestAction([[seg(7)], [k]])
    assert action.token_length() == 3
    out = action.get_result(embedding)
    assert out.shape == (1, 3, EMBED_DIM)
    # First match should be the token itself.
    assert torch.allclose(out[0, 0], embedding.weight[7])


def test_lerp_is_registered_as_avg_alias():
    lerp_cls = get_action_by_name("lerp")
    avg_cls = get_action_by_name("avg")
    assert issubclass(lerp_cls, avg_cls)


def test_token_lengths():
    a, b = seg(1, 2, 3), seg(4)
    assert ProjectAction([[a], [b]]).token_length() == 3
    assert RejectAction([[a], [b]]).token_length() == 3
    assert RenormAction([[a], [b]]).token_length() == 3
    std = PromptSegment(text="0.1", tokens=[1])
    assert NoiseAction([[a], [std]]).token_length() == 3
