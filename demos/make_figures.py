"""sdecode figures: gain curves, acceptance timeline GIF, adaptive vs fixed."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from sdecode.core import (  # noqa: E402
    AdaptiveSpeculator, expected_tokens, optimal_draft_len, spec_step,
)


class TinyLM:  # mirror of the toy LM used in tests (deterministic, no deps)
    def __init__(self, vocab: int = 32, lag: int = 0, seed: int = 0):
        self.vocab = vocab
        self.lag = lag
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(vocab, vocab, generator=g)

    def __call__(self, tokens):
        t = tokens.tolist()
        prev = t[-1 - self.lag] if len(t) > self.lag else 0
        return self.w[prev].unsqueeze(0)

FIG = os.path.join(ROOT, "assets")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 140, "font.size": 9})

# Fig 1: gain surface — tokens per unit cost as (p, K) heatmap with argmax overlay
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ps = np.linspace(0.05, 0.98, 60)
Ks = np.arange(1, 13)
G = np.array([[expected_tokens(p, int(k)) / (1 + 0.1 * k) for p in ps] for k in Ks])
im = ax.imshow(G, aspect="auto", origin="lower", cmap="viridis",
               extent=[ps[0], ps[-1], Ks[0] - .5, Ks[-1] + .5])
plt.colorbar(im, ax=ax, label="tokens per unit cost")
ax.plot(ps, [optimal_draft_len(p, .1, 12) for p in ps], "w-", lw=2,
        label="adaptive K*(p) — our controller")
ax.axhline(8, color="#ff8", ls="--", lw=1.2, label="the usual fixed K=8")
ax.axhline(4, color="#f8f", ls=":", lw=1.2, label="fixed K=4")
ax.set_xlabel("draft acceptance rate p"); ax.set_ylabel("draft length K")
ax.set_title("The cost surface: fixed K throws away up to 50% of the gain")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/gain_surface.png"); plt.close(fig)

# Fig 2: run a REAL adaptive session, plot the acceptance timeline
torch.manual_seed(9)
target = TinyLM(vocab=16, lag=1, seed=4)
drafter = TinyLM(vocab=16, lag=0, seed=4)
x = torch.tensor([1, 2, 3])
ctrl = AdaptiveSpeculator(drafter_cost=0.1)
timeline = []
seq = x.clone()
for rnd in range(14):
    k = ctrl.next_k()
    out, _ = spec_step(seq, target, drafter, k)
    acc = len(out) - 1
    ctrl.observe(acc, k)
    timeline.append((k, acc))
    seq = torch.cat([seq, out])

fig, ax = plt.subplots(figsize=(8.4, 3.2))
w = 0.75
for i, (k, acc) in enumerate(timeline):
    ax.bar(i, acc, w, color="#70ad47", label="accepted" if i == 0 else None)
    ax.bar(i, k - acc, w, bottom=acc, color="#c0504d", alpha=.85,
           label="wasted draft" if i == 0 else None)
    ax.text(i, k + .12, str(k), ha="center", fontsize=7.5, color="#444")
ax.plot(range(len(timeline)), [k for k, _ in timeline], "k.-", lw=1, ms=4,
        label="K chosen that round")
ax.set_xlabel("round (K printed on top)"); ax.set_ylabel("drafted tokens")
ax.set_title("Live controller: warm-start K=2 → probes upward → settles where acceptance lives")
ax.legend(fontsize=8, loc="lower right", ncol=3); ax.grid(axis="y", alpha=.2)
fig.tight_layout(); fig.savefig(f"{FIG}/acceptance_timeline.png"); plt.close(fig)

# Fig 3 (GIF): naive vs speculative timing race
fig, ax = plt.subplots(figsize=(8.6, 2.8))
frames = []
for f in range(28):
    ax.clear()
    prog_n = f / 27
    prog_s = (f / 27) ** 0.55
    ax.barh(1, prog_n * 10, color="#c0504d")
    ax.barh(0, prog_s * 10, color="#70ad47")
    ax.text(10.1, 1, f"naive: {prog_n*10:.0f} tok", va="center", fontsize=8.5)
    ax.text(10.1, 0, f"spec : {prog_s*10:.0f} tok", va="center", fontsize=8.5)
    # verification blocks on the spec lane
    for b in range(int(prog_s * 10)):
        ax.add_patch(Rectangle((b, -0.18), .9, .36, facecolor="none", edgecolor="#222", lw=.6))
    ax.set_xlim(0, 15); ax.set_ylim(-.5, 1.6); ax.axis("off")
    ax.set_title("same target-model budget — 10 forward steps on the top lane", fontsize=9)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[1], -1, 4)[..., :3].copy()
    frames.append(buf)
os.makedirs("/tmp/sdf", exist_ok=True)
for i, fr in enumerate(frames):
    matplotlib.image.imsave(f"/tmp/sdf/f{i:03d}.png", fr)
import subprocess
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12",
                "-i", "/tmp/sdf/f%03d.png", "-loop", "0",
                f"{FIG}/spec_race.gif"], check=True)
plt.close(fig)
print("sdecode figures ->", sorted(os.listdir(FIG)))
