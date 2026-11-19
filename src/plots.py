"""Figures for the report: (1) dev sweep, (2) test results per condition with 95% bootstrap CIs."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from common import RESULTS

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed categorical order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "white"})

# (1) dev sweep: primary metric and EN guardrail vs alpha, one line per layer
d = pd.read_csv(RESULTS / "dev_sweep.csv")
base = d[d.layer.isna()].iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
for ax, col, title in [(axes[0], "sv:nordic_sent_share", "No/da sentence share, dev SV prompts (lower is better)"),
                       (axes[1], "en:en_answer_rate", "EN prompts answered in English (guardrail)")]:
    ax.axhline(base[col], color=MUTED, lw=1, ls="--", label="baseline")
    for c, (L, g) in zip(SERIES, d[d.layer.notna()].groupby("layer")):
        ax.plot(g.alpha, g[col], "-o", color=c, lw=2, ms=5, label=f"layer {int(L)}")
    ax.set_xscale("log", base=2); ax.set_xlabel("α (multiples of ‖d‖)"); ax.set_title(title, fontsize=9, color=INK)
axes[1].axhline(0.9, color="#e34948", lw=1, ls=":", label="guardrail 0.90")
axes[0].legend(frameon=False, fontsize=8); axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(RESULTS / "fig_dev_sweep.png", dpi=200)

# (2) test: primary metric per condition with CI
s = json.load(open(RESULTS / "test_summary.json"))["conditions"]
names = [c["condition"] for c in s]
m = [c["sv:nordic_sent_share"] for c in s]
lo = [c["sv:nordic_sent_share"] - c["sv:nordic_sent_ci"][0] for c in s]
hi = [c["sv:nordic_sent_ci"][1] - c["sv:nordic_sent_share"] for c in s]
fig, ax = plt.subplots(figsize=(5.5, 2.8))
y = range(len(names))[::-1]
cols = [SERIES[0] if n == "edit_nordic" else MUTED for n in names]
ax.errorbar(m, list(y), xerr=[lo, hi], fmt="none", ecolor=cols, elinewidth=2, capsize=0)
ax.scatter(m, list(y), c=cols, s=40, zorder=3)
for yi, mi in zip(y, m):
    ax.annotate(f"{mi:.1%}", (mi, yi), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=INK)
ax.set_yticks(list(y)); ax.set_yticklabels(names); ax.grid(axis="y", visible=False)
ax.set_xlabel("Mean per-answer share of no/da sentences (test SV prompts, 95% CI)")
fig.tight_layout(); fig.savefig(RESULTS / "fig_test.png", dpi=200)
print("saved figures")
