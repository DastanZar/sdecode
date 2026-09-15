import sys

import torch

sys.path.insert(0, ".")
from sdecode.core import (  # noqa: E402
    AdaptiveSpeculator,
    adaptive_speculative_generate,
    expected_tokens,
    greedy_verify,
    optimal_draft_len,
    spec_step,
)


class TinyLM:
    """Deterministic toy LM: next token = (weighted sum of input) % vocab.
    `lag` shifts the pattern so the drafter (lag=0) approximates the target (lag=1)."""

    def __init__(self, vocab: int = 32, lag: int = 0, noise: float = 0.0, seed: int = 0):
        self.vocab = vocab
        self.lag = lag
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(vocab, vocab, generator=g) * (1 + noise)

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.tolist()
        prev = t[-1 - self.lag] if len(t) > self.lag else 0
        logits = self.w[prev]
        return logits.unsqueeze(0)  # [1, vocab] for last position


def test_greedy_verify_accepts_correct_prefix():
    torch.manual_seed(0)
    # draft [5, 9, 2]; target agrees on 5,9 but wants 7 at pos 2
    logits = torch.zeros(4, 10)
    logits[0, 5] = 5
    logits[1, 9] = 5
    logits[2, 7] = 5
    logits[3, 1] = 5
    out = greedy_verify(logits, torch.tensor([5, 9, 2]))
    assert out.tolist() == [5, 9, 7], out


def test_greedy_verify_full_accept():
    logits = torch.zeros(3, 10)
    logits[0, 4] = 1
    logits[1, 6] = 1
    logits[2, 8] = 1
    out = greedy_verify(logits, torch.tensor([4, 6]))
    assert out.tolist() == [4, 6]


def test_output_identical_to_target_greedy():
    """The core theorem: speculative output == plain target greedy decode."""
    torch.manual_seed(1)
    target = TinyLM(vocab=16, lag=1, seed=3)
    drafter = TinyLM(vocab=16, lag=0, seed=3)
    x = torch.tensor([1, 2, 3])
    # naive greedy from target
    seq = x.clone()
    with torch.no_grad():
        for _ in range(12):
            nxt = int(target(seq)[-1].argmax())
            seq = torch.cat([seq, torch.tensor([nxt])])
    # speculative
    out, _ = spec_step(x, target, lambda t: target(torch.cat([t, t[-1:] * 0])), 0) if False else (None, None)
    res = adaptive_speculative_generate(x, target, drafter, n_rounds=8)
    spec_seq = res["sequence"]
    # speculative emits at least as many tokens as rounds; compare the shared prefix
    n = min(len(spec_seq), len(seq))
    assert torch.equal(spec_seq[:n], seq[:n]), (spec_seq, seq)


def test_expected_tokens_formula():
    assert abs(expected_tokens(0.5, 1) - 1.5) < 1e-9
    assert abs(expected_tokens(0.5, 3) - 1.875) < 1e-9
    assert expected_tokens(1.0, 5) == 6.0


def test_optimal_draft_len_high_acceptance():
    # with p=0.9 and cheap drafter, longer drafts pay off
    k = optimal_draft_len(0.9, 0.1, k_max=8)
    assert k >= 4, k
    # with p=0.3, drafting is mostly wasted
    k_low = optimal_draft_len(0.3, 0.1, k_max=8)
    assert k_low <= 2, k_low


def test_adaptive_controller_converges_to_optimal():
    ctrl = AdaptiveSpeculator(drafter_cost=0.1)
    # feed mostly-accepted rounds (p ~ 0.8)
    for _ in range(10):
        ctrl.observe(8, 8)
    assert ctrl.next_k() >= 3
    # crash the acceptance rate
    for _ in range(10):
        ctrl.observe(0, 4)
    assert ctrl.next_k() <= 2


def test_spec_beats_naive_on_tokens_per_target_call():
    torch.manual_seed(2)
    target = TinyLM(vocab=16, lag=0, seed=5)        # drafter == target: perfect acceptance
    drafter = TinyLM(vocab=16, lag=0, seed=5)
    x = torch.tensor([0])
    res = adaptive_speculative_generate(x, target, drafter, n_rounds=10)
    assert res["effective_tokens_per_target_call"] > 1.0
    assert res["target_model_calls"] < res["naive_target_calls_for_same_tokens"]
