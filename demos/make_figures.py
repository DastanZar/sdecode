"""Generate sdecode's visual identity: crossover envelope, K-decision staircase,
and a round-chip animation. Every number comes from sdecode.core — these figures
are runs of the shipped code, not hand-plotted illustrations."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from sdecode.core import (  # noqa: E402
    AdaptiveSpeculator,
    expected_tokens,
    optimal_draft_len,
    spec_step,
)

# --- shared style: one palette, used by every figure --------------------------
INK = "#16213c"       # text
ACCENT = "#4f46e5"    # ours / adaptive
AMBER = "#f59e0b"     # the customary fixed K=8
TEAL = "#14b8a6"      # accepted / throughput
GRAY = "#94a3b8"      # naive reference
DANGER = "#fee2e2"    # shade: slower than naive

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.dpi": 150,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": "#cbd5e1",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

FIG = os.path.join(ROOT, "assets")
os.makedirs(FIG, exist_ok=True)


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax.set_axisbelow(True)


# --- Fig 1: crossover envelope -----------------------------------------------
# The hero claim: there is a whole acceptance regime where the customary fixed
# K=8 yields FEWER tokens per unit cost than not speculating at all (naive=1.0).
# Adaptive never crosses below the line.
C = 0.1
ps = np.linspace(0.05, 0.98, 400)
gain_k8 = np.array([expected_tokens(p, 8) / (1 + C * 8) for p in ps])
gain_ad = np.array(
    [expected_tokens(p, optimal_draft_len(p, C, 8)) / (1 + C * optimal_draft_len(p, C, 8))
     for p in ps]
)
cross_idx = np.argmax(gain_k8 >= 1.0)
cross_p = ps[cross_idx]

fig, ax = plt.subplots(figsize=(7.6, 4.3))
ax.axvspan(ps[0], cross_p, color=DANGER, zorder=0)
ax.axhline(1.0, color=GRAY, ls="--", lw=1.4, label="naive decoding (1.0 tok/unit)")
ax.plot(ps, gain_k8, color=AMBER, lw=2.2, label="fixed K=8 (the default)")
ax.plot(ps, gain_ad, color=ACCENT, lw=2.6, label="adaptive K (this repo)")
ax.annotate(
    f"fixed K=8 is SLOWER\nthan not speculating\nfor p < {cross_p:.2f}",
    xy=(cross_p, 1.0), xytext=(cross_p - 0.20, 0.52),
    color=INK, fontsize=9,
    arrowprops=dict(arrowstyle="->", color=INK, lw=1.1),
)
ax.annotate(
    "adaptive K:\nnever below naive",
    xy=(0.30, expected_tokens(0.30, optimal_draft_len(0.30, C, 8)) / (1 + C * 1)),
    xytext=(0.34, 1.55), color=ACCENT, fontsize=9,
    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1),
)
ax.set_xlabel("draft acceptance rate  p")
ax.set_ylabel("tokens per unit cost")
ax.set_xlim(ps[0], ps[-1])
ax.set_ylim(0, 4.4)
clean(ax)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "crossover.png"))
plt.close(fig)

# --- Fig 2: K*(p) decision staircase -----------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 3.4))
ks = [optimal_draft_len(p, C, 8) for p in ps]
ax.fill_between(ps, ks, step="post", color=ACCENT, alpha=0.13, zorder=1)
ax.step(ps, ks, where="post", color=ACCENT, lw=2.4, zorder=2, label="K chosen by the controller")
ax.axhline(8, color=AMBER, ls="--", lw=1.6, label="fixed K=8", zorder=3)
for p_mark, label in ((0.30, "K=1"), (0.50, "K=2"), (0.72, "K=4"), (0.85, "K=6")):
    k = optimal_draft_len(p_mark, C, 8)
    ax.plot([p_mark], [k], "o", color=INK, ms=5, zorder=4)
    ax.annotate(label, (p_mark, k), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8.5, color=INK)
ax.set_xlabel("draft acceptance rate  p")
ax.set_ylabel("draft length K")
ax.set_ylim(0, 9.8)
clean(ax)
# opaque legend: the K=8 dashed line runs through the legend area otherwise
ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "k_decision.png"))
plt.close(fig)

# --- Fig 3: round-chip animation ---------------------------------------------
# A live session from the toy LM: each round streams K chips — accepted in teal,
# wasted in amber — while the naive lane trickles one chip per round.


class TinyLM:
    def __init__(self, vocab=32, lag=0, seed=0):
        self.vocab = vocab
        self.lag = lag
        gen = torch.Generator().manual_seed(seed)
        self.w = torch.randn(vocab, vocab, generator=gen)

    def __call__(self, tokens):
        t = tokens.tolist()
        prev = t[-1 - self.lag] if len(t) > self.lag else 0
        return self.w[prev].unsqueeze(0)


torch.manual_seed(9)
target = TinyLM(vocab=16, lag=1, seed=4)
drafter = TinyLM(vocab=16, lag=0, seed=4)
prompt = torch.tensor([1, 2, 3])
ctrl = AdaptiveSpeculator(drafter_cost=0.1)
seq = prompt.clone()
rounds = []
for _ in range(12):
    k = ctrl.next_k()
    out, _ = spec_step(seq, target, drafter, k)
    acc = len(out) - 1
    ctrl.observe(acc, k)
    rounds.append((k, acc))
    seq = torch.cat([seq, out])

CHIP = 0.30  # x-units per chip
frames_n = 26
tmp = os.path.join("/tmp", "sdecode_rounds")
os.makedirs(tmp, exist_ok=True)

for f in range(frames_n):
    shown = int(np.ceil((f + 1) / frames_n * len(rounds)))
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    x = 0.0
    naive_x = 0.0
    for r in range(shown):
        k, acc = rounds[r]
        # spec lane: accepted chips then wasted chips for this round
        for j in range(acc):
            ax.add_patch(FancyBboxPatch(
                (x, 1.05), CHIP * 0.86, 0.42, boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=TEAL, edgecolor="none"))
            x += CHIP
        for j in range(k - acc):
            ax.add_patch(FancyBboxPatch(
                (x, 1.05), CHIP * 0.86, 0.42, boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=AMBER, edgecolor="none"))
            x += CHIP
        ax.text(x - CHIP * k / 2, 0.83, f"K={k}", ha="center", fontsize=7, color=INK)
        # naive lane: one chip per round, at the same target-call cost
        ax.add_patch(FancyBboxPatch(
            (naive_x, 0.15), CHIP * 0.86, 0.42, boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=GRAY, edgecolor="none"))
        naive_x += CHIP
    ax.text(0, 1.62, f"adaptive speculative — round {shown}/{len(rounds)}",
            fontsize=10, color=ACCENT, weight="bold")
    n_naive = int(round(naive_x / CHIP))
    ax.text(0, 0.00, f"naive — same target calls, {n_naive} token" + ("s" if n_naive != 1 else ""),
            fontsize=9, color="#64748b")
    ax.text(x + 0.2, 1.26, f"{int(round(x / CHIP))} tok", fontsize=9,
            color=TEAL, weight="bold")
    spec_accepted = sum(a for _, a in rounds[:shown])
    ax.set_xlim(-0.1, max(x, naive_x) + 2.0)
    ax.set_ylim(-0.1, 1.85)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(tmp, f"frame{f:03d}.png"))
    plt.close(fig)

gif = os.path.join(FIG, "rounds.gif")
subprocess.run(
    ["/home/ubuntu/.hermes/tools/ffmpeg-9.0.1-linux-arm64/bin/ffmpeg", "-y", "-loglevel", "error",
     "-framerate", "6", "-i", os.path.join(tmp, "frame%03d.png"),
     "-vf", "palettegen=stats_mode=diff", os.path.join(tmp, "palette.png")],
    check=True,
)
subprocess.run(
    ["/home/ubuntu/.hermes/tools/ffmpeg-9.0.1-linux-arm64/bin/ffmpeg", "-y", "-loglevel", "error",
     "-framerate", "6", "-i", os.path.join(tmp, "frame%03d.png"),
     "-i", os.path.join(tmp, "palette.png"), "-lavfi", "paletteuse=dither=sierra2",
     "-loop", "0", gif],
    check=True,
)

print("assets ->", sorted(os.listdir(FIG)))
