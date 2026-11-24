"""Tables of the report, from the result files in this repository.

    python report/tables.py

Writes report/generated/*.tex. Each number is the median over the 3 training
seeds. Standard library only.
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "generated"


def methods_table():
    rows = defaultdict(lambda: defaultdict(dict))  # method -> seed -> mass -> row
    with open(ROOT / "results" / "d3" / "h2_h3_test.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["method"]][int(r["training_seed"])][int(r["payload_mass_g"])] = {k: float(v) for k, v in r.items() if k != "method"}
    methods = ["P0c", "P1", "P2", "P3"]

    def med(method, f):
        return median(f(rows[method][s]) for s in sorted(rows[method]))

    in_range = lambda s: sum(s[m]["rmse_xy"] for m in (20, 40, 60, 80, 100)) / 5  # noqa: E731
    lines = [
        ("Fall rate at 100 g", lambda s: s[100]["fall_rate"], "%.3f"),
        ("Fall rate at 150 g", lambda s: s[150]["fall_rate"], "%.3f"),
        ("Fall rate at 200 g", lambda s: s[200]["fall_rate"], "%.3f"),
        ("RMSE at 0 g (m/s)", lambda s: s[0]["rmse_xy"], "%.3f"),
        ("RMSE, 20--100 g (m/s)", in_range, "%.3f"),
        ("Velocity change, backward (m/s)", lambda s: s[100]["vx_backward"] - s[0]["vx_backward"], "%.3f"),
        ("Velocity change, forward fast (m/s)", lambda s: s[100]["vx_forward_fast"] - s[0]["vx_forward_fast"], "%.3f"),
    ]
    tex = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule", " & " + " & ".join(methods) + "\\\\", "\\midrule"]
    for i, (name, f, fmt) in enumerate(lines):
        if i in (3, 5):
            tex.append("\\midrule")
        tex.append(name + " & " + " & ".join(fmt % med(m, f) for m in methods) + "\\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "methods.tex").write_text("\n".join(tex) + "\n")


def monitor_tables():
    mon = defaultdict(list)   # (method, monitor, mass, quantity) -> values of the training seeds
    fall = defaultdict(list)  # (method, mode, mass) -> values
    with open(ROOT / "results" / "d4" / "h5_h6_test.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["value"] in ("", "nan"):
                continue
            if r["table"] == "monitor":
                mon[(r["method"], r["row"], r["payload_mass_g"], r["quantity"])].append(float(r["value"]))
            elif r["quantity"] == "fall_rate":
                fall[(r["method"], r["row"], r["payload_mass_g"])].append(float(r["value"]))

    signals = ["neck", "residual", "value", "probe"]
    quantities = [("False-alarm rate", "0-100", "false_alarm_rate", "%.3f"), ("Falls caught", "200", "caught", "%.2f"),
                  ("Lead time (s)", "200", "median_lead_s", "%.2f"), ("Delay (s)", "200", "median_delay_s", "%.2f")]
    tex = ["\\begin{tabular}{@{}llcccc@{}}", "\\toprule", " & & " + " & ".join(signals) + "\\\\", "\\midrule"]
    for method in ("P1", "P3"):
        for i, (name, mass, q, fmt) in enumerate(quantities):
            cells = " & ".join(fmt % median(mon[(method, s + "_plain", mass, q)]) for s in signals)
            tex.append((method if i == 0 else "") + " & " + name + " & " + cells + "\\\\")
        if method == "P1":
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "monitors.tex").write_text("\n".join(tex) + "\n")

    modes = [("none", "none"), ("residual", "residual, stand"), ("onset", "stand at payload"), ("neck_drop", "neck, drop"), ("residual_drop", "residual, drop")]
    tex = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
           " & \\multicolumn{2}{c}{P1} & \\multicolumn{2}{c}{P3}\\\\", "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
           "Monitor, fallback & 150 g & 200 g & 150 g & 200 g\\\\", "\\midrule"]
    for mode, name in modes:
        tex.append(name + " & " + " & ".join("%.3f" % median(fall[(m, mode, g)]) for m in ("P1", "P3") for g in ("150", "200")) + "\\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "fallback.tex").write_text("\n".join(tex) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    methods_table()
    monitor_tables()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.tex"))))
