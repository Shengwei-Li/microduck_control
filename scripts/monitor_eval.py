"""Offline evaluation of the runtime monitors on stored episodes (D4).

    python monitor_eval.py --fit <a.npz> --cal <b.npz> --eval <c.npz> [<d.npz> ...] \\
        --dynamics <model>.pt [--kappa 1.5] [--alpha 0.05] [--name P1_seed0] [--out <file>.json]

The files come from monitor_collect.py, all for the same policy:
  --fit   the scales of each signal come from these episodes (the dynamics model
          and the probe were also fitted on them)
  --cal   the thresholds come from these episodes (split conformal)
  --eval  the monitors are scored on these episodes
Only episodes with a payload inside the training range (0 to 100 g) are
"normal" and are used for --fit and --cal.

Signals: neck (position error of the neck joint), residual (learned one-step
model), value (minus the critic output; simulation only), probe (payload mass
from the linear probe). Each signal gives two monitors: a plain threshold, and
a CUSUM with allowance kappa.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from payload_duck import monitor  # noqa: E402
from payload_duck.dynamics import Dynamics, residual_signal, valid_frames  # noqa: E402

MAX_NORMAL_G = 100.0
SIGNALS = ("neck", "residual", "value", "probe")


class Episodes:
    """The stored episodes of one or more files, with the four signals (steps, robots)."""

    def __init__(self, paths, dynamics, device):
        files = [np.load(p) for p in paths]
        f0 = files[0]
        self.dt = float(f0["step_dt"])
        self.first = int(f0["first_step"])
        self.onset = round(float(f0["onset_s"]) / self.dt) - self.first  # index in the stored window
        self.mass = np.concatenate([f["mass_g"] for f in files])
        self.command = np.concatenate([f["command"] for f in files])
        self.fall = np.concatenate([f["fall_step"] for f in files])  # step in the episode, -1 = no fall
        terms = monitor.term_slices(f0["term_names"], f0["term_dims"])
        sig = {k: [] for k in SIGNALS}
        for f in files:
            frame = f["frame"]
            valid = valid_frames(len(frame), self.first, f["done_step"])
            nan = np.where(valid, 0.0, np.nan)
            sig["neck"].append(monitor.neck_error(frame, terms, f["joint_names"]).astype(np.float64) + nan)
            sig["residual"].append(residual_signal(dynamics, frame, f["action"], valid, device))
            sig["value"].append(-f["value"].astype(np.float64) + nan)
            sig["probe"].append(f["mass_est"].astype(np.float64) + nan)
        self.signal = {k: np.concatenate(v, axis=1) for k, v in sig.items()}
        self.normal = self.mass <= MAX_NORMAL_G


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fit", required=True)
    p.add_argument("--cal", required=True)
    p.add_argument("--eval", nargs="+", required=True)
    p.add_argument("--dynamics", required=True)
    p.add_argument("--kappa", type=float, default=1.5)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--name", default="")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dyn = Dynamics()
    dyn.load_state_dict(torch.load(args.dynamics, map_location="cpu"))
    fit, cal, ev = (Episodes(paths, dyn, args.device) for paths in ([args.fit], [args.cal], args.eval))

    result = {"name": args.name, "kappa": args.kappa, "alpha": args.alpha, "monitors": {}}
    fall_idx = np.where(ev.fall >= 0, ev.fall - ev.first, -1)  # index in the stored window
    masses = sorted(set(ev.mass.tolist()))
    print(f"{args.name}: kappa {args.kappa}, target false-alarm rate {args.alpha}, {int(ev.normal.sum())} normal and "
          f"{int((~ev.normal).sum())} other episodes")
    print("monitor          false alarm | " + " | ".join(f"{m:.0f} g: alarm, delay, falls, caught, lead" for m in masses if m > MAX_NORMAL_G))
    for name in SIGNALS:
        # Scales from the normal fit episodes, from the time when the payload appears.
        scale = monitor.Standardizer.fit(fit.signal[name][fit.onset:, fit.normal])
        for kind, kappa in (("plain", None), ("cusum", args.kappa)):
            score = lambda e: monitor.cusum(scale(e.signal[name]), kappa)  # noqa: E731
            thr = monitor.conformal_threshold(monitor.episode_max(score(cal)[:, cal.normal]), args.alpha)
            alarm = monitor.first_alarm(score(ev), thr)  # index in the stored window, -1 = none
            rows = {}
            for m in masses:
                k = ev.mass == m
                a, f = alarm[k], fall_idx[k]
                fell = f >= 0
                caught = fell & (a >= 0) & (a < f)
                after = a[a >= ev.onset]
                rows[f"{m:.0f}"] = {
                    "episodes": int(k.sum()),
                    "alarm_rate": float((a >= 0).mean()),
                    "alarm_before_payload": float(((a >= 0) & (a < ev.onset)).mean()),
                    "median_delay_s": float(np.median(after - ev.onset) * ev.dt) if len(after) else None,
                    "falls": int(fell.sum()),
                    "caught": float(caught.sum() / fell.sum()) if fell.any() else None,
                    "median_lead_s": float(np.median((f - a)[caught]) * ev.dt) if caught.any() else None,
                }
            false_alarm = float((alarm[ev.normal] >= 0).mean())
            result["monitors"][f"{name}_{kind}"] = {"mu": scale.mu, "sigma": scale.sigma, "threshold": thr,
                                                    "false_alarm_rate": false_alarm, "by_mass": rows}
            cells = []
            for m in masses:
                if m > MAX_NORMAL_G:
                    r = rows[f"{m:.0f}"]
                    fmt = lambda v, s: "   - " if v is None else s % v  # noqa: E731
                    cells.append("%.2f %s %4d %s %s" % (r["alarm_rate"], fmt(r["median_delay_s"], "%5.2f"), r["falls"],
                                                       fmt(r["caught"], "%5.2f"), fmt(r["median_lead_s"], "%5.2f")))
            print(f"{name + '_' + kind:16s} {false_alarm:11.3f} | " + " | ".join(cells))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
