"""Real-model engine harness.

The core module is the reference implementation: it verifies one drafted
position per target call so the semantics stay easy to read. A real engine
verifies the whole draft with ONE target forward pass — feed x + draft and
read the K+1 logit rows that cover every draft position plus the bonus slot.
This module provides that batched path over Hugging Face causal LMs, together
with the naive and fixed-K baselines, so the receipts in receipts/real_run.json
are produced by the shipped code on real weights — not a simulation.

Interface contract (must match core.spec_step exactly):

    batched_verify_logits = logits over x+draft, last K+1 rows
    greedy_verify(batched_verify_logits, draft) == core.greedy_verify output

The last K+1 rows are exactly the per-position target predictions for draft
positions 0..K (row -j predicts position n+... see greedy_verify), so the
acceptance decision is identical to the reference path; only the number of
forward passes changes.
"""

from __future__ import annotations

import time

from typing import Protocol

import torch

from sdecode.core import AdaptiveSpeculator, greedy_verify


class LogitsEngine(Protocol):
    """Structural type: any engine exposing these two calls works."""

    def last_logits(self, ids: torch.Tensor, n: int) -> torch.Tensor: ...

    def next_token(self, ids: torch.Tensor) -> int: ...


class HFEngine:
    """Adapter: a HF causal LM with a next-token-logits interface."""

    def __init__(self, model_name: str, num_threads: int | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if num_threads is not None:
            torch.set_num_threads(num_threads)
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def encode(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt").input_ids[0]

    def decode(self, ids) -> str:
        return self.tok.decode(ids)

    @torch.no_grad()
    def last_logits(self, ids: torch.Tensor, n: int) -> torch.Tensor:
        """Logits for the last n positions of a forward pass over `ids`.

        Row j (from the end) predicts the token after ids[:-j] — i.e. rows
        ids[-1], ids[-2], ... The engine uses n = K+1 for batched verification.
        """
        return self.model(ids.unsqueeze(0)).logits[0][-n:]

    @torch.no_grad()
    def next_token(self, ids: torch.Tensor) -> int:
        return int(self.last_logits(ids, 1)[-1].argmax())


def measure_drafter_cost(target: HFEngine, drafter: HFEngine, n: int = 6) -> float:
    """Measured drafter-vs-target forward cost ratio on this box.

    The shipped default (c=0.1) assumes a drafter ~10x cheaper than the target
    — true for a small drafter on a GPUserver, not for two similar-sized
    models on CPU. Timed here with warm-up so the controller optimizes for
    the machine it is actually running on. Same BPE family assumed (the
    harness encodes with the target tokenizer and feeds the drafter).
    """
    sample = target.encode(
        "The quick brown fox jumps over the lazy dog. " * 3
    )[:32]
    with torch.no_grad():
        for _ in range(2):  # warm-up
            target.last_logits(sample, 1)
            drafter.last_logits(sample, 1)
        t0 = time.perf_counter()
        for _ in range(n):
            target.last_logits(sample, 1)
        t_target = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n):
            drafter.last_logits(sample, 1)
        t_drafter = time.perf_counter() - t0
    return t_drafter / t_target


def spec_round_batched(seq: torch.Tensor, target: LogitsEngine, drafter: LogitsEngine, k: int):
    """One speculative round with batched verification.

    Returns dict with emitted tokens, draft tokens, accepted count, and the
    number of target forwards consumed (always 1).
    """
    draft = []
    cur = seq
    for _ in range(k):
        nxt = drafter.next_token(cur)
        draft.append(nxt)
        cur = torch.cat([cur, torch.tensor([nxt])])
    draft_t = torch.tensor(draft)
    tgt_rows = target.last_logits(torch.cat([seq, draft_t]), k + 1)
    out = greedy_verify(tgt_rows, draft_t)
    return {
        "emitted": out,
        "draft": draft_t,
        "accepted": len(out) - 1,
        "k": k,
        "target_forwards": 1,
    }


def naive_generate(seq: torch.Tensor, target: LogitsEngine, n_tokens: int) -> dict:
    """One target forward per token — the no-speculation baseline."""
    t0 = time.perf_counter()
    out = []
    cur = seq
    for _ in range(n_tokens):
        nxt = target.next_token(cur)
        out.append(nxt)
        cur = torch.cat([cur, torch.tensor([nxt])])
    dt = time.perf_counter() - t0
    return {
        "tokens": out,
        "n_tokens": n_tokens,
        "target_forwards": n_tokens,
        "drafter_forwards": 0,
        "wall_s": dt,
        "ms_per_token": dt / n_tokens * 1e3,
        "tokens_per_target_forward": 1.0,
        "rounds": [],
    }


def _run_rounds(seq, target, drafter, pick_k, max_tokens: int, observe=None) -> dict:
    """Shared loop for adaptive / fixed-K lanes.

    pick_k(round_index) -> K for this round; observe(accepted, k) is called
    after every round so an adaptive controller tracks the live session.
    """
    t0 = time.perf_counter()
    cur = seq
    rounds = []
    target_forwards = 0
    drafter_forwards = 0
    while (len(cur) - len(seq)) < max_tokens:
        rt0 = time.perf_counter()
        k = pick_k(len(rounds))
        r = spec_round_batched(cur, target, drafter, k)
        cur = torch.cat([cur, r["emitted"]])
        target_forwards += 1
        drafter_forwards += k
        if observe is not None:
            observe(r["accepted"], k)
        rounds.append(
            {
                "round": len(rounds) + 1,
                "k": k,
                "accepted": r["accepted"],
                "emitted": r["emitted"].tolist(),
                "draft": r["draft"].tolist(),
                "wall_ms": (time.perf_counter() - rt0) * 1e3,
            }
        )
    dt = time.perf_counter() - t0
    n_tokens = len(cur) - len(seq)
    return {
        "tokens": cur[len(seq):].tolist(),
        "n_tokens": n_tokens,
        "rounds": rounds,
        "target_forwards": target_forwards,
        "drafter_forwards": drafter_forwards,
        "wall_s": dt,
        "ms_per_token": dt / n_tokens * 1e3,
        "tokens_per_target_forward": n_tokens / max(1, target_forwards),
    }


def adaptive_generate(
    seq: torch.Tensor,
    target: HFEngine,
    drafter: HFEngine,
    max_tokens: int,
    drafter_cost: float = 0.1,
) -> dict:
    """Adaptive-K lane: core.AdaptiveSpeculator picks K every round, live."""
    ctrl = AdaptiveSpeculator(drafter_cost=drafter_cost)
    res = _run_rounds(
        seq,
        target,
        drafter,
        lambda i: ctrl.next_k(),
        max_tokens,
        observe=lambda accepted, k: ctrl.observe(accepted, k),
    )
    res["k_choices"] = [r["k"] for r in res["rounds"]]
    res["p_estimate"] = ctrl.estimate_p()
    return res


def fixed_k_generate(
    seq: torch.Tensor, target: HFEngine, drafter: HFEngine, max_tokens: int, k: int = 8
) -> dict:
    """Fixed-K lane — the shipped default being audited."""
    return _run_rounds(seq, target, drafter, lambda i: k, max_tokens)
