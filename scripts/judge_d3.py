"""Tests of H2 and H3 (prereg/D2_hypotheses.md) on the test seeds.

For each policy, the script pools the episodes of the test seeds (per_env.csv).
For each method, it takes the median over the 3 training seeds.

    python judge_d3.py [--seeds 2000 2001] [--out results/d3/h2_h3_test]

The last table (change of velocity with mass) is exploratory. It is not part
of the preregistration.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MASSES = [0, 20, 40, 60, 80, 100, 120, 150, 200]
IN_RANGE = [20, 40, 60, 80, 100]
POLICIES = {  # method -> result folders of the 3 training seeds
    "P0c": ["p0c_seed0", "p0c_seed1", "p0c_seed2"],
    "P1": ["p1_seed0_pilot", "p1_seed1", "p1_seed2"],
    "P2": ["p2_seed0", "p2_seed1", "p2_seed2"],
    "P3": ["p3_seed0", "p3_seed1", "p3_seed2"],
}


def read_policy(name, seeds, split):
    """Return {(mass_g, command): (fell, n_valid, rmse_xy, mean_vx)} arrays, pooled over the seeds."""
    rows = defaultdict(list)
    for seed in seeds:
        path = ROOT / "results" / "eval" / name / f"model_3998_{split}{seed}" / "per_env.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[(int(float(r["payload_mass_g"])), r["command"])].append(
                    (r["fell"] == "True", float(r["n_valid_steps"]), float(r["rmse_xy"]), float(r["mean_vx"])))
    return {k: np.nan_to_num(np.array(v, dtype=float)) for k, v in rows.items()}


def metrics(groups):
    """Per mass: fall rate and pooled RMSE xy over all commands, and mean vx of two commands."""
    out = {}
    for m in MASSES:
        a = np.concatenate([v for (mass, _), v in groups.items() if mass == m])
        n = a[:, 1]
        out[m] = {
            "episodes": len(a),
            "fall_rate": a[:, 0].mean(),
            "rmse_xy": np.sqrt((n * a[:, 2] ** 2).sum() / n.sum()),  # Pool the squared errors over all valid steps.
        }
        for c in ("backward", "forward_fast"):
            g = groups[(m, c)]
            out[m]["vx_" + c] = (g[:, 1] * g[:, 3]).sum() / g[:, 1].sum()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[2000, 2001])
    p.add_argument("--out", default=str(ROOT / "results" / "d3" / "h2_h3_test"))
    p.add_argument("--split", default="test", help="'dev' with --seeds 1000 repeats the development numbers.")
    args = p.parse_args()

    M = {method: [metrics(read_policy(n, args.seeds, args.split)) for n in names] for method, names in POLICIES.items()}

    def per_seed(method, f):
        return np.array([f(s) for s in M[method]])

    def med(method, f):
        return float(np.median(per_seed(method, f)))

    def show(title, f, fmt="%.3f"):
        print(title)
        for method in POLICIES:
            v = per_seed(method, f)
            print("  %-4s median " % method + fmt % np.median(v) + "   seeds " + " ".join(fmt % x for x in v))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "training_seed", "payload_mass_g", "episodes", "fall_rate", "rmse_xy", "vx_backward", "vx_forward_fast"])
        for method in POLICIES:
            for s, pm in enumerate(M[method]):
                for m in MASSES:
                    w.writerow([method, s, m, pm[m]["episodes"]] + [round(float(pm[m][k]), 6) for k in ("fall_rate", "rmse_xy", "vx_backward", "vx_forward_fast")])

    print("episodes per mass and policy:", M["P1"][0][0]["episodes"])
    for m in MASSES:
        show("fall rate at %d g" % m, lambda s, m=m: s[m]["fall_rate"])
    show("RMSE xy at 0 g", lambda s: s[0]["rmse_xy"], "%.4f")
    in_range = lambda s: np.mean([s[m]["rmse_xy"] for m in IN_RANGE])
    show("RMSE xy, mean over 20-100 g", in_range, "%.4f")

    print("\nVerdicts (median over 3 training seeds)")
    worst = max(med("P1", lambda s, m=m: s[m]["fall_rate"]) for m in MASSES if m <= 100)
    print("H2a  highest fall rate of P1 up to 100 g: %.4f (limit 0.05) -> %s" % (worst, "holds" if worst <= 0.05 else "fails"))
    r1, r0 = med("P1", lambda s: s[0]["rmse_xy"]), med("P0c", lambda s: s[0]["rmse_xy"])
    print("H2b  RMSE at 0 g: P1 %.4f, P0c %.4f, ratio %.3f (limit 1.05) -> %s" % (r1, r0, r1 / r0, "holds" if r1 >= 1.05 * r0 else "fails"))
    e1, e2, e3 = med("P1", in_range), med("P2", in_range), med("P3", in_range)
    gap = e1 - e2
    if gap <= 0:
        print("H3a  RMSE 20-100 g: P1 %.4f, P2 %.4f, P3 %.4f. P2 is not better than P1: there is no gap to close -> not supported" % (e1, e2, e3))
    else:
        frac = (e1 - e3) / gap
        print("H3a  RMSE 20-100 g: P1 %.4f, P2 %.4f, P3 %.4f. P3 closes %.2f of the gap (limit 0.5) -> %s" % (e1, e2, e3, frac, "holds" if frac >= 0.5 else "fails"))
    f1, f3 = med("P1", lambda s: s[150]["fall_rate"]), med("P3", lambda s: s[150]["fall_rate"])
    print("H3b  fall rate at 150 g: P1 %.4f, P3 %.4f -> %s" % (f1, f3, "holds" if f3 < f1 else "fails"))

    print("\nExploratory (not preregistered): change of mean vx from 0 g to 100 g")
    for c in ("backward", "forward_fast"):
        show("  " + c, lambda s, c=c: s[100]["vx_" + c] - s[0]["vx_" + c])
    print("Wrote", out.with_suffix(".csv"))


if __name__ == "__main__":
    main()
