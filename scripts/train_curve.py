"""Turn a training log into a CSV file and a plot of the learning curves.

Usage (from the upstream folder, so that it uses the upstream environment):
    cd ~/projectD/third_party/microduck_rl
    uv run python /mnt/d/.../scripts/train_curve.py LOG_FILE OUT_PREFIX

Writes OUT_PREFIX.csv (one row per iteration) and OUT_PREFIX.png.
Prints the values at some iterations and the means over windows of 250 iterations.
"""

import csv
import re
import sys

import matplotlib

matplotlib.use("Agg")  # No window: write the plot to a file.
import matplotlib.pyplot as plt

# Label in the log -> column name in the CSV.
FIELDS = {
    "Mean reward": "reward",
    "Mean episode length": "episode_length",
    "Episode_Termination/fell_over": "fell_over",
    "Episode_Termination/time_out": "time_out",
    "Metrics/twist/error_vel_xy": "error_vel_xy",
    "Metrics/twist/error_vel_yaw": "error_vel_yaw",
    "Episode_Reward/track_linear_velocity": "r_track_lin",
    "Episode_Reward/action_rate_l2": "r_action_rate",
    "Curriculum/action_rate_weight": "cur_action_rate_weight",
    "Curriculum/standing_envs": "cur_standing_envs",
    "Mean action std": "action_std",
    "Steps per second": "steps_per_s",
}


def parse(path):
    rows, row = [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"Learning iteration (\d+)/", line)
            if m:
                if row:
                    rows.append(row)
                row = {"iteration": int(m.group(1))}
                continue
            if row is None or ":" not in line:
                continue
            label, _, value = line.strip().rpartition(":")
            label = label.strip()
            if label in FIELDS:
                try:
                    row[FIELDS[label]] = float(value)
                except ValueError:
                    pass
    if row:
        rows.append(row)
    return rows


def main():
    log, prefix = sys.argv[1], sys.argv[2]
    rows = parse(log)
    cols = ["iteration"] + list(FIELDS.values())
    with open(prefix + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} iterations")
    show = ["reward", "episode_length", "fell_over", "error_vel_xy", "error_vel_yaw",
            "r_action_rate", "cur_action_rate_weight", "cur_standing_envs", "action_std"]
    print("window      " + " ".join(f"{c[:14]:>14}" for c in show))
    for start in range(0, len(rows), 250):
        win = rows[start:start + 250]
        means = [sum(r.get(c, 0.0) for r in win) / len(win) for c in show]
        print(f"{start:4d}-{start + len(win) - 1:4d}   " + " ".join(f"{v:14.4f}" for v in means))

    # Four panels: reward, falls, tracking error, curriculum.
    it = [r["iteration"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        (ax[0, 0], ["reward"], "mean reward"),
        (ax[0, 1], ["fell_over", "time_out"], "episode ends per iteration"),
        (ax[1, 0], ["error_vel_xy", "error_vel_yaw"], "velocity tracking error"),
        (ax[1, 1], ["cur_action_rate_weight", "cur_standing_envs"], "curriculum"),
    ]
    for a, keys, title in panels:
        for k in keys:
            a.plot(it, [r.get(k, float("nan")) for r in rows], label=k, linewidth=0.8)
        a.set_title(title)
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    for a in ax[1]:
        a.set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(prefix + ".png", dpi=120)
    print("wrote", prefix + ".csv", prefix + ".png")


if __name__ == "__main__":
    main()
