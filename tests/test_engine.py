"""Batched engine path must be decision-identical to the reference path.

core.spec_step verifies one drafted position per target call; engine's
spec_round_batched verifies with a single forward. Both feed the same
greedy_verify, so for identical models and drafts the emitted tokens must be
exactly equal. These tests pin that contract with fake batched LMs — no
transformers download involved.
"""

import torch

from sdecode.engine import spec_round_batched
from sdecode.core import greedy_verify, spec_step


class TinyLM:
    """Deterministic tiny LM fixture: next-token logits = w[last-lag token]."""

    def __init__(self, vocab: int, lag: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(vocab, vocab, generator=g)
        self.lag = lag

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.tolist()
        prev = t[-1 - self.lag] if len(t) > self.lag else 0
        return self.w[prev].unsqueeze(0)


class FakeBatchLM:
    """Duck-typed batched engine over the same weights as TinyLM(lag=0).

    last_logits(ids, n): next-token logits for the last n positions.
    With lag=0, the prediction after prefix ids[:t+1] is w[ids[t]] — so the
    last n rows are w[ids[t]] for t = len-n .. len-1.
    """

    def __init__(self, tinylm: TinyLM):
        self.w = tinylm.w

    @torch.no_grad()
    def last_logits(self, ids: torch.Tensor, n: int) -> torch.Tensor:
        return torch.stack([self.w[int(t)] for t in ids[len(ids) - n:]])

    @torch.no_grad()
    def next_token(self, ids: torch.Tensor) -> int:
        return int(self.last_logits(ids, 1)[-1].argmax())


def test_batched_round_emits_identical_tokens():
    torch.manual_seed(0)
    target_ref = TinyLM(vocab=16, lag=0, seed=3)
    drafter_ref = TinyLM(vocab=16, lag=0, seed=9)
    target = FakeBatchLM(target_ref)
    drafter = FakeBatchLM(drafter_ref)

    x = torch.tensor([1, 2, 3])
    for k in (1, 2, 5, 8):
        ref, _ = spec_step(x, target_ref, drafter_ref, k)
        bat = spec_round_batched(x, target, drafter, k)
        assert bat["emitted"].tolist() == ref.tolist(), f"k={k}"
        assert bat["accepted"] == len(ref) - 1
        assert bat["target_forwards"] == 1


def test_batched_verify_rows_match_reference_stack():
    """The K+1 rows the batched path reads must equal the rows the reference
    path stacks position-by-position."""
    torch.manual_seed(1)
    ref = TinyLM(vocab=16, lag=0, seed=4)
    eng = FakeBatchLM(ref)
    x = torch.tensor([5, 7])
    k = 4
    draft = []
    cur = x
    for _ in range(k):
        nxt = int(ref(cur).argmax())
        draft.append(nxt)
        cur = torch.cat([cur, torch.tensor([nxt])])
    draft_t = torch.tensor(draft)
    stacked = torch.stack([ref(torch.cat([x, draft_t[:i]]))[-1] for i in range(k + 1)])
    batched = eng.last_logits(torch.cat([x, draft_t]), k + 1)
    assert torch.equal(stacked, batched)
    # and greedy_verify agrees on both representations
    assert torch.equal(greedy_verify(stacked, draft_t), greedy_verify(batched, draft_t))
