"""Tests of H5 and H6 (prereg/D4_hypotheses.md).

    python judge_d4.py --offline <folder> --fallback <folder> [--seeds 2000 2001] [--out results/d4/h5_h6_test]

--offline holds <method>_seed<k>.json from monitor_eval.py, scored on the
pooled test seeds. --fallback holds <method>_seed<k>_eval<seed>_<mode>.json
from fallback_eval.py. For each method, the script takes the median over the
3 training seeds. Fall rates are the mean over the evaluation seeds (they have
the same number of episodes).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("P1", "P3")
SIGNALS = ("neck", "residual", "value", "probe")
MODES = ("none", "residual", "onset", "neck_drop", "residual_drop")
MASSES = ("0", "20", "40", "60", "80", "100", "120", "150", "200")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--offline", required=True)
    p.add_argument("--fallback", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[2000, 2001])
    p.add_argument("--out", default=str(ROOT / "results" / "d4" / "h5_h6_test"))
    args = p.parse_args()
    off_dir, fb_dir = Path(args.offline), Path(args.fallback)

    off = {m: [json.loads((off_dir / f"{m}_seed{t}.json").read_text())["monitors"] for t in range(3)] for m in METHODS}

    def monitor_value(method, signal, mass, key, kind="plain"):
        """Per training seed: one value of a monitor at one mass (None -> nan)."""
        v = [s[f"{signal}_{kind}"]["by_mass"][mass][key] for s in off[method]]
        return np.array([np.nan if x is None else x for x in v], dtype=float)

    def false_alarm(method, signal, kind="plain"):
        return np.array([s[f"{signal}_{kind}"]["false_alarm_rate"] for s in off[method]])

    def fall(method, mode, mass, key="fall_rate"):
        """Per training seed: mean over the evaluation seeds."""
        return np.array([np.mean([json.loads((fb_dir / f"{method}_seed{t}_eval{e}_{mode}.json").read_text())["by_mass"][mass][key]
                                  for e in args.seeds]) for t in range(3)])

    med = np.nanmedian
    show = lambda v, fmt="%.3f": fmt % med(v) + " (" + "/".join(fmt % x for x in v) + ")"  # noqa: E731

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "method", "training_seed", "row", "payload_mass_g", "quantity", "value"])
        for m in METHODS:
            for t in range(3):
                for s in SIGNALS:
                    for kind in ("plain", "cusum"):
                        w.writerow(["monitor", m, t, f"{s}_{kind}", "0-100", "false_alarm_rate", round(float(false_alarm(m, s, kind)[t]), 4)])
                        for mass in MASSES:
                            for key in ("alarm_rate", "median_delay_s", "falls", "caught", "median_lead_s"):
                                w.writerow(["monitor", m, t, f"{s}_{kind}", mass, key, monitor_value(m, s, mass, key, kind)[t]])
                for mode in MODES:
                    for mass in MASSES:
                        w.writerow(["fallback", m, t, mode, mass, "fall_rate", round(float(fall(m, mode, mass)[t]), 4)])
                for s in ("neck", "residual"):
                    for mass in MASSES:
                        w.writerow(["fallback", m, t, "none", mass, f"alarm_rate_{s}", round(float(fall(m, "none", mass, f"alarm_rate_{s}")[t]), 4)])

    for m in METHODS:
        print(f"\n{m}: monitors at 200 g, median (training seeds 0/1/2)")
        for s in SIGNALS:
            print(f"  {s:9s} false alarm {show(false_alarm(m, s))} | caught {show(monitor_value(m, s, '200', 'caught'), '%.2f')} | "
                  f"lead {show(monitor_value(m, s, '200', 'median_lead_s'), '%.2f')} | delay {show(monitor_value(m, s, '200', 'median_delay_s'), '%.2f')}")
        print(f"{m}: fall rate")
        for mass in ("120", "150", "200"):
            print(f"  {mass:>3s} g: " + " | ".join(f"{mode} {show(fall(m, mode, mass))}" for mode in MODES))
        print(f"{m}: alarm rate of the residual monitor by mass: " + " ".join(f"{mass}:{med(fall(m, 'none', mass, 'alarm_rate_residual')):.3f}" for mass in MASSES))

    print("\nVerdicts (median over 3 training seeds; a hypothesis for both methods holds only if it holds for both)")

    def verdict(name, text, per_method):
        ok = all(v for v, _ in per_method.values())
        print(f"{name}  {text} -> {'holds' if ok else 'fails'}")
        for m, (v, detail) in per_method.items():
            print(f"       {m}: {detail} -> {'holds' if v else 'fails'}")

    r = {m: max(med(false_alarm(m, s)) for s in SIGNALS) for m in METHODS}
    verdict("H5a", "false-alarm rate of each monitor at most 0.08", {m: (r[m] <= 0.08, "highest %.3f" % r[m]) for m in METHODS})
    r = {m: (med(monitor_value(m, "residual", "200", "caught")), med(monitor_value(m, "residual", "200", "median_lead_s"))) for m in METHODS}
    verdict("H5b", "residual at 200 g: caught at least 0.95 and lead time above 0.3 s",
            {m: (c >= 0.95 and lead > 0.3, "caught %.3f, lead %.2f s" % (c, lead)) for m, (c, lead) in r.items()})
    r = {m: (med(monitor_value(m, "neck", "200", "caught")), med(monitor_value(m, "residual", "200", "caught"))) for m in METHODS}
    verdict("H5c", "neck at 200 g: caught lower than residual by at least 0.10",
            {m: (b - a >= 0.10, "neck %.3f, residual %.3f" % (a, b)) for m, (a, b) in r.items()})
    r = {m: (med(fall(m, "residual", "200")), med(fall(m, "none", "200"))) for m in METHODS}
    verdict("H5d", "residual + stand at 200 g: fall rate more than half of no fallback",
            {m: (a > 0.5 * b, "%.3f against %.3f" % (a, b)) for m, (a, b) in r.items()})
    r = {m: (med(fall(m, "residual_drop", "200")), med(fall(m, "none", "200"))) for m in METHODS}
    verdict("H5e", "residual + drop at 200 g: fall rate at most a quarter of no fallback",
            {m: (a <= 0.25 * b, "%.3f against %.3f" % (a, b)) for m, (a, b) in r.items()})
    a, b = med(fall("P3", "neck_drop", "150")), med(fall("P3", "none", "150"))
    verdict("H5f", "P3 at 150 g: neck + drop has a higher fall rate than no fallback", {"P3": (a > b, "%.3f against %.3f" % (a, b))})
    a, b = med(monitor_value("P1", "value", "200", "median_delay_s")), med(monitor_value("P1", "residual", "200", "median_delay_s"))
    verdict("H6a", "P1 at 200 g: value has a lower delay than residual", {"P1": (a < b, "%.2f s against %.2f s" % (a, b))})
    a, b = med(monitor_value("P3", "probe", "200", "caught")), med(monitor_value("P3", "residual", "200", "caught"))
    verdict("H6b", "P3 at 200 g: probe catches less than residual", {"P3": (a < b, "%.3f against %.3f" % (a, b))})
    print("\nWrote", out.with_suffix(".csv"))


if __name__ == "__main__":
    main()
