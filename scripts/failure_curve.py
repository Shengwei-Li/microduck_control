"""Failure curve: fall rate and mean velocity against the payload mass.

Reads per_env.csv from one or more evaluation folders (for example two seeds),
pools the episodes, and writes <out>.csv (one row per mass and command, and
one row "ALL" per mass) and <out>.png.

    python failure_curve.py --runs <folder> [<folder> ...] --out <path without extension>
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COMMAND_VX = {"forward_slow": 0.2, "forward_fast": 0.4, "backward": -0.3}  # Commands shown in the velocity plot.


def read_runs(folders):
    """Return {(mass_g, command): list of rows} with numeric fields, pooled over the folders."""
    groups = defaultdict(list)
    for folder in folders:
        with open(Path(folder) / "per_env.csv", newline="") as f:
            for row in csv.DictReader(f):
                groups[(float(row["payload_mass_g"]), row["command"])].append({
                    "fell": row["fell"] == "True",
                    "fall_time_s": float(row["fall_time_s"]),
                    "n": int(row["n_valid_steps"]),
                    "mean_vx": float(row["mean_vx"]),
                    "mean_wz": float(row["mean_wz"]),
                })
    return groups


def summarize(rows):
    """Fall rate, median fall time, and mean velocity (weighted by the valid steps of each episode)."""
    fell = np.array([r["fell"] for r in rows])
    n = np.array([r["n"] for r in rows], dtype=float)
    times = np.array([r["fall_time_s"] for r in rows])[fell]

    def weighted(key):
        x = np.nan_to_num(np.array([r[key] for r in rows]))
        return float((n * x).sum() / n.sum()) if n.sum() > 0 else float("nan")

    return {
        "episodes": len(rows),
        "fall_rate": float(fell.mean()),
        "median_fall_time_s": float(np.median(times)) if len(times) else float("nan"),
        "mean_vx": weighted("mean_vx"),
        "mean_wz": weighted("mean_wz"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    args = p.parse_args()

    groups = read_runs(args.runs)
    masses = sorted({m for m, _ in groups})
    commands = list(dict.fromkeys(c for _, c in groups))  # Keep the file order.
    table = {}
    for m in masses:
        for c in commands:
            table[(m, c)] = summarize(groups[(m, c)])
        table[(m, "ALL")] = summarize([r for c in commands for r in groups[(m, c)]])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["episodes", "fall_rate", "median_fall_time_s", "mean_vx", "mean_wz"]
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["payload_mass_g", "command"] + keys)
        for (m, c), s in table.items():
            w.writerow([m, c] + [round(s[k], 4) for k in keys])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for c in commands:
        ax1.plot(masses, [table[(m, c)]["fall_rate"] for m in masses], marker=".", lw=1, label=c)
    ax1.plot(masses, [table[(m, "ALL")]["fall_rate"] for m in masses], color="k", marker="o", lw=2, label="all commands")
    ax1.set_xlabel("payload mass (g)")
    ax1.set_ylabel("fall rate")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    for i, (c, cmd) in enumerate(COMMAND_VX.items()):
        color = f"C{i}"
        ax2.plot(masses, [table[(m, c)]["mean_vx"] for m in masses], marker=".", color=color, label=c)
        ax2.axhline(cmd, color=color, ls="--", lw=0.8)
    ax2.set_xlabel("payload mass (g)")
    ax2.set_ylabel("mean forward velocity (m/s)")
    ax2.set_title("dashed lines: commands", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    if args.title:
        fig.suptitle(args.title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)

    print(f"{'mass':>5s} " + " ".join(f"{c[:9]:>9s}" for c in commands + ["ALL"]) + " | t_fall  vx_ffast  vx_back")
    for m in masses:
        falls = " ".join(f"{table[(m, c)]['fall_rate']:9.3f}" for c in commands + ["ALL"])
        print(f"{m:5.0f} {falls} | {table[(m, 'ALL')]['median_fall_time_s']:6.2f} "
              f"{table[(m, 'forward_fast')]['mean_vx']:9.3f} {table[(m, 'backward')]['mean_vx']:8.3f}")
    print(f"Wrote {out.with_suffix('.csv')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
