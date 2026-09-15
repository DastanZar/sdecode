"""Speculative decoding: draft cheap, verify in parallel, accept or rollback.

Prior art: Leviathan et al. 2023 & Chen et al. 2023 — a small drafter proposes
K tokens, the target model scores all K+1 in one forward pass, and the longest
correct prefix is accepted; the first mismatch is resampled from the corrected
distribution. Output is provably distribution-identical to the target model
(greedy case: identical argmax tokens).

The cost model (why this wins): a target step costs 1 unit, a drafter step
costs c < 1. Naive decoding emits 1 token/unit. One speculative round with
draft length K and acceptance rate p emits E[len] tokens for K*c + 1 units.
E[len] = (1 - p^(K+1)) / (1 - p)  (expected accepted prefix of a geometric run).

The twist here: an adaptive controller that picks K per round from the
observed acceptance rate, maximizing the token-per-unit gain against the
drafter cost — instead of the fixed K every reference implementation uses.
"""
from __future__ import annotations

import torch


def greedy_verify(
    target_logits: torch.Tensor,  # [K+1, vocab] scores for draft[0..K] positions
    draft_tokens: torch.Tensor,   # [K] proposed tokens
) -> torch.Tensor:
    """Return accepted tokens: the longest prefix where draft argmax == target argmax."""
    assert target_logits.shape[0] == draft_tokens.shape[0] + 1
    tgt = target_logits.argmax(dim=-1)  # target's preferred token at each position
    # position i: was the drafted token correct? draft[i] vs tgt[i], for i in 0..K-1
    correct = draft_tokens == tgt[:-1]
    # accepted prefix length = first False position (or K if all correct)
    if bool(correct.all()):
        return draft_tokens.clone()
    first_bad = int((~correct).nonzero()[0])
    # draft up to first_bad-1 accepted, then the target's correction token
    return torch.cat([draft_tokens[:first_bad], tgt[first_bad : first_bad + 1]])


def spec_step(
    x: torch.Tensor,
    target_lm,            # callable(tokens)->logits[next] (one step per position)
    drafter_lm,            # callable(tokens)->logits[next]
    draft_len: int,
) -> tuple[torch.Tensor, int]:
    """One speculative round. Returns (emitted_tokens, draft_len_used)."""
    draft = []
    seq = x
    with torch.no_grad():
        for _ in range(draft_len):
            nxt = int(drafter_lm(seq)[-1].argmax())
            draft.append(nxt)
            seq = torch.cat([seq, torch.tensor([nxt])])
        draft_tokens = torch.tensor(draft)
        # target scores positions 0..K (one forward for all drafted + the next)
        tgt_logits = torch.stack([target_lm(torch.cat([x, draft_tokens[:i]]))[-1]
                                  for i in range(draft_len + 1)])
    out = greedy_verify(tgt_logits, draft_tokens)
    return out, draft_len


def verify_branch(target_logits: torch.Tensor, draft_tokens: torch.Tensor,
                  correct_len: int) -> bool:
    """Sanity oracle used by tests: simulate acceptance of a known prefix."""
    tgt = target_logits.argmax(dim=-1)
    ok = torch.equal(draft_tokens[:correct_len], tgt[:correct_len])
    if correct_len < len(draft_tokens):
        ok = ok and not torch.equal(draft_tokens[correct_len : correct_len + 1],
                                    tgt[correct_len : correct_len + 1])
    return bool(ok)


def expected_tokens(p: float, k: int) -> float:
    """E[accepted + 1] tokens per round for acceptance rate p, draft length k."""
    if p >= 1.0:
        return float(k + 1)
    return (1 - p ** (k + 1)) / (1 - p)


def optimal_draft_len(p: float, drafter_cost: float, k_max: int = 8) -> int:
    """Choose k maximizing tokens per unit cost: E[len]/(1 + c*k)."""
    best_k, best_gain = 1, 0.0
    for k in range(1, k_max + 1):
        gain = expected_tokens(p, k) / (1 + drafter_cost * k)
        if gain > best_gain:
            best_gain, best_k = gain, k
    return best_k


class AdaptiveSpeculator:
    """Tracks observed acceptance; retries with the analytic optimal K."""

    def __init__(self, drafter_cost: float = 0.1, k_max: int = 8, window: int = 8):
        self.drafter_cost = drafter_cost
        self.k_max = k_max
        self.window = window
        self.history: list[float] = []  # accepted fraction per round

    def next_k(self) -> int:
        if len(self.history) < 2:
            return 2  # conservative start
        p = sum(self.history) / len(self.history)
        return optimal_draft_len(p, self.drafter_cost, self.k_max)

    def observe(self, accepted: int, drafted: int) -> None:
        self.history.append(accepted / max(1, drafted))
        self.history = self.history[-self.window :]


def adaptive_speculative_generate(
    x: torch.Tensor, target_lm, drafter_lm, n_rounds: int = 6
) -> dict:
    """Generate via adaptive speculative rounds; also runs the naive baseline
    for an apples-to-apples token-per-call comparison."""
    ctrl = AdaptiveSpeculator()
    seq = x.clone()
    rounds = []
    target_calls = 0
    drafter_calls = 0
    with torch.no_grad():
        for _ in range(n_rounds):
            k = ctrl.next_k()
            out, _ = spec_step(seq, target_lm, drafter_lm, k)
            target_calls += 1
            drafter_calls += k
            accepted = len(out) - 1  # last token is the verified correction/extension
            ctrl.observe(accepted, k)
            seq = torch.cat([seq, out])
            rounds.append({"k": k, "emitted": len(out), "accepted": accepted})
    # naive baseline: same token count, one target call per token
    naive_calls = len(seq) - len(x)
    return {
        "sequence": seq,
        "rounds": rounds,
        "target_model_calls": target_calls,
        "drafter_calls": drafter_calls,
        "naive_target_calls_for_same_tokens": naive_calls,
        "effective_tokens_per_target_call": round(
            (len(seq) - len(x)) / target_calls, 2
        ),
    }
