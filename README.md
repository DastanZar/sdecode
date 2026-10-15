# sdecode

![ci](https://github.com/DastanZar/sdecode/actions/workflows/acceptance.yml/badge.svg)

Speculative decoding ships a draft length someone chose once, on a machine that
isn't yours. `sdecode` measures the acceptance rate you're actually getting and
re-answers the question every round — with a closed form, not a search.

**[Open the live tuner →](app/tuner.html)** — drag your measured acceptance and
drafter cost, watch the answer move. Same math as
`sdecode.core.optimal_draft_len`, ported line-for-line.

---

## The default is a bet, and it can lose to doing nothing

A drafter proposes K tokens, the target model verifies them in one pass, the
matching prefix survives. Everyone ships K=8. But the value of K=8 depends on
your acceptance rate — and below about **p ≈ 0.45**, hard-coded K=8 returns
*fewer tokens per unit cost than simply not speculating*:

![crossover envelope](assets/crossover.png)

That shaded region isn't a slowdown you tolerate; it's the default losing to
naive decoding. At p=0.3 the receipt measures K=8 at **0.794 tok/unit** while
not speculating gives 1.0. The adaptive line never dips below the floor — it
answers K=1 there and takes the win.

## What the controller decides

One argmax over integers, fed by an 8-round window of observed acceptance.
No training, no bandit, no tuning knob — the objective is closed-form:

```
E(K, p) = (1 − p^(K+1)) / (1 − p)        expected accepted run
gain     = E(K, p) / (1 + c·K)            c = drafter/target cost ratio
K*       = argmax over integer K ∈ [1, 8]
```

![K decision staircase](assets/k_decision.png)

## One session, watched

Twelve rounds through a toy LM where the drafter tracks the target closely but
not perfectly. Green chips are accepted draft tokens, amber are rejected. Same
target-model calls on both lanes:

![rounds](assets/rounds.gif)

## Receipts

`python demos/demo.py` writes `receipts/receipt.json` — every number below
comes from it, measured, reproducible:

| acceptance p | controller picks | adaptive tok/unit | fixed K=8 | adaptive vs K=8 |
|---|---|---|---|---|
| 0.30 | K=1 | 1.182 | 0.794 | **+48.9%** |
| 0.50 | K=2 | 1.458 | 1.109 | **+31.5%** |
| 0.70 | K=4 | 1.981 | 1.777 | **+11.5%** |
| 0.80 | K=6 | 2.470 | 2.405 | +2.7% |
| 0.90 | K=8 | 3.403 | 3.403 | +0.0% |
| 0.95 | K=8 | 4.108 | 4.108 | +0.0% |

At p=0.8, c=0.1: **2.47 tokens per target-model call** against naive's 1.0 —
and the same sweep shows the adaptive choice never trailing the best fixed
candidate from {2, 4, 8} (it ties where the fixed set already contains the
optimum, and picks values the set can't express).

## The property everything rests on

A speedup that changes your output distribution isn't a speedup, it's a
quantizer wearing a benchmark's hat. Greedy verification is provably
distribution-identical to the target model, and the suite pins it empirically:
same seed, naive greedy vs speculative, token-for-token equality
(`test_output_identical_to_target_greedy`).

## Try it

```bash
pip install -e ".[dev]"
pytest -q              # 7 tests: formula, controller convergence, identity
python demos/demo.py   # receipt + figures
python demos/make_figures.py
```

## Prior art

The acceptance criterion and corrected-distribution sampling are Leviathan et
al. ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192), ICML 2023) and Chen
et al. ([arXiv:2302.01318](https://arxiv.org/abs/2302.01318)). This repo is
built on that; the delta is the adaptive-K controller, its break-even analysis
against naive decoding, and the measurement apparatus around both — the math
was already complete, the packaging of the decision is the work here.
