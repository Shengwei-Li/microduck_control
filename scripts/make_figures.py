"""Figures of the test results (D3 and D4), from the result files in this repository.

    python make_figures.py

Writes results/d3/methods_test.png, results/d3/probe_test.png and
results/d4/monitors_test.png. Lines are the median over the 3 training seeds.
Bands and error bars are the lowest and the highest training seed.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
MASSES = [0, 20, 40, 60, 80, 100, 120, 150, 200]

# One fixed colour and marker for each policy, in all figures.
STYLE = {
    "P0": ("#2a78d6", "o", "P0: no payload in training"),
    "P0c": ("#eb6834", "s", "P0c: more training, no payload"),
    "P1": ("#1baf7a", "^", "P1: random payload"),
    "P2": ("#eda100", "D", "P2: knows the mass"),
    "P3": ("#e87ba4", "v", "P3: observation history"),
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5,
})


def training_range(ax, x, m):
    """Mark the training range of the payload mass (0 to 100 g)."""
    ax.axvspan(0, 100, color="#f0efec", zorder=0)
    ax.text(50, m, "training range of P1, P2, P3", ha="center", va="top", fontsize=8, color=MUTED)


def band(ax, x, values, key, label=True):
    """Median line with a band from the lowest to the highest training seed. values: (seeds, len(x))."""
    color, marker, name = STYLE[key]
    v = np.asarray(values, dtype=float)
    if len(v) > 1:
        ax.fill_between(x, v.min(0), v.max(0), color=color, alpha=0.18, linewidth=0)
    ax.plot(x, np.median(v, 0), color=color, marker=marker, markeredgecolor=SURFACE, markeredgewidth=0.8,
            label=name if label else None)


def methods_figure():
    table = defaultdict(lambda: defaultdict(dict))  # method -> seed -> mass -> row
    with open(R / "d3" / "h2_h3_test.csv", newline="") as f:
        for r in csv.DictReader(f):
            table[r["method"]][int(r["training_seed"])][int(r["payload_mass_g"])] = r
    p0 = {}
    with open(R / "d2" / "p0_failure_curve_test.csv", newline="") as f:
        for r in csv.DictReader(f):
            p0[(int(float(r["payload_mass_g"])), r["command"])] = r

    def series(method, key):
        return [[float(table[method][s][m][key]) for m in MASSES] for s in sorted(table[method])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9))
    training_range(ax1, MASSES, 0.97)
    band(ax1, MASSES, [[float(p0[(m, "ALL")]["fall_rate"]) for m in MASSES]], "P0")
    for method in ("P0c", "P1", "P2", "P3"):
        band(ax1, MASSES, series(method, "fall_rate"), method)
    ax1.set(xlabel="payload mass (g)", ylabel="fall rate, all commands", ylim=(-0.02, 1.02), title="Falls")
    ax1.legend(loc="center left", fontsize=8)

    training_range(ax2, MASSES, 0.085)
    band(ax2, MASSES, [[float(p0[(m, "backward")]["mean_vx"]) for m in MASSES]], "P0", label=False)
    for method in ("P0c", "P1", "P2", "P3"):
        band(ax2, MASSES, series(method, "vx_backward"), method, label=False)
    ax2.axhline(-0.3, color=MUTED, linestyle="--", linewidth=1)
    ax2.text(200, -0.292, "command: -0.3 m/s", ha="right", va="bottom", fontsize=8, color=MUTED)
    ax2.set(xlabel="payload mass (g)", ylabel="mean forward velocity (m/s)", ylim=(-0.32, 0.1),
            title="Backward walking: velocity against payload")
    fig.suptitle("Test seeds 2000 and 2001, 768 episodes per mass and policy. Line: median of 3 training seeds. Band: lowest to highest seed.",
                 fontsize=8, color=MUTED, y=0.995)
    fig.tight_layout()
    fig.savefig(R / "d3" / "methods_test.png", dpi=160)
    plt.close(fig)


def probe_figure():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for method in ("P1", "P3"):
        runs = [json.loads((R / "d3" / "probe" / f"{method}_seed{s}.json").read_text())["r2_by_time"] for s in range(3)]
        t = np.array([float(k) for k in runs[0]])
        v = np.array([[r[k] for k in runs[0]] for r in runs])
        keep = (t >= 4.0) & (t <= 12.0)
        band(ax, t[keep], np.clip(v[:, keep], -0.25, 1.0), method)
    ax.axvline(5.0, color=MUTED, linestyle="--", linewidth=1)
    ax.text(5.08, 0.98, "payload appears", va="top", fontsize=8, color=MUTED)
    ax.axhline(0.7, color=MUTED, linestyle=":", linewidth=1)
    ax.text(12.0, 0.69, "H4: 0.7", ha="right", va="top", fontsize=8, color=MUTED)
    ax.set(xlabel="time in the episode (s)", ylabel="R$^2$ of the linear probe", ylim=(-0.27, 1.0),
           title="Payload mass from the last hidden layer of the actor")
    ax.legend(loc="center right", fontsize=8)
    fig.text(0.01, 0.005, "Fit on development seed 1000, scored on test seeds 2000 and 2001. Values below -0.25 are drawn at -0.25.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(R / "d3" / "probe_test.png", dpi=160)
    plt.close(fig)


def monitors_figure():
    fall = defaultdict(dict)   # (method, seed, mode) -> mass -> value
    alarm = defaultdict(dict)  # (method, seed) -> mass -> alarm rate of the residual monitor
    with open(R / "d4" / "h5_h6_test.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["table"] != "fallback":
                continue
            key = (r["method"], int(r["training_seed"]))
            if r["quantity"] == "fall_rate":
                fall[key + (r["row"],)][int(r["payload_mass_g"])] = float(r["value"])
            elif r["quantity"] == "alarm_rate_residual":
                alarm[key][int(r["payload_mass_g"])] = float(r["value"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9))
    modes = [("none", "no fallback"), ("residual", "residual\nmonitor,\nstand"), ("onset", "stand when\nthe payload\nappears"),
             ("neck_drop", "neck\nmonitor,\ndrop"), ("residual_drop", "residual\nmonitor,\ndrop")]
    width = 0.36
    for i, method in enumerate(("P1", "P3")):
        color = STYLE[method][0]
        v = np.array([[fall[(method, s, mode)][200] for mode, _ in modes] for s in range(3)])
        x = np.arange(len(modes)) + (i - 0.5) * (width + 0.03)
        mid = np.median(v, 0)
        ax1.bar(x, mid, width, color=color, label=STYLE[method][2])
        ax1.errorbar(x, mid, yerr=[mid - v.min(0), v.max(0) - mid], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        for xi, m, hi in zip(x, mid, v.max(0)):
            ax1.text(xi, hi + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax1.set_xticks(np.arange(len(modes)), [label for _, label in modes], fontsize=8)
    ax1.set(ylabel="fall rate at 200 g", ylim=(0, 0.82), title="Falls at 200 g with each fallback")
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="upper right", fontsize=8)

    training_range(ax2, MASSES, 0.97)
    for method in ("P1", "P3"):
        color, marker, name = STYLE[method]
        a = np.array([[alarm[(method, s)][m] for m in MASSES] for s in range(3)])
        f_ = np.array([[fall[(method, s, "none")][m] for m in MASSES] for s in range(3)])
        ax2.plot(MASSES, np.median(a, 0), color=color, marker=marker, markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax2.plot(MASSES, np.median(f_, 0), color=color, marker=marker, markeredgecolor=SURFACE, markeredgewidth=0.8,
                 linestyle=(0, (2, 2)), linewidth=1.5)
    ax2.plot([], [], color=MUTED, label="alarm rate of the residual monitor")
    ax2.plot([], [], color=MUTED, linestyle=(0, (2, 2)), linewidth=1.5, label="fall rate without fallback")
    ax2.text(152, 0.70, "P1", color=INK, fontsize=8, ha="left")
    ax2.text(128, 0.80, "P3", color=INK, fontsize=8, ha="right")
    ax2.set(xlabel="payload mass (g)", ylabel="fraction of episodes", ylim=(-0.02, 1.02), title="The monitor stops more robots than would fall")
    ax2.legend(loc="center left", fontsize=8)
    fig.suptitle("Test seeds 2000 and 2001. Median of 3 training seeds. Error bars: lowest to highest seed. Target false-alarm rate 0.05.",
                 fontsize=8, color=MUTED, y=0.995)
    fig.tight_layout()
    fig.savefig(R / "d4" / "monitors_test.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    methods_figure()
    probe_figure()
    monitors_figure()
    print("wrote results/d3/methods_test.png, results/d3/probe_test.png, results/d4/monitors_test.png")
