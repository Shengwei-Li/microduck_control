"""Linear probe for the payload mass (H4, prereg/D2_hypotheses.md).

    python probe_fit.py --fit <a.npz> --score <b.npz> [<c.npz> ...] [--name P3_seed0] [--out <file>.json]

The files come from probe_collect.py. The script fits a linear model
(least squares, with intercept) from the 128 hidden values to the payload
mass in grams. It fits on the episodes of --fit and gives R^2 on the episodes
of --score. One sample is one robot at one time.

Frozen settings (see prereg/D3_probe_settings.md): masses from 0 to 100 g,
samples from 2 s after the payload appears, robots that have not fallen.
The script also gives R^2 for each sample time (exploratory).
"""

import argparse
import json
from pathlib import Path

import numpy as np

MAX_MASS_G = 100.0  # Training range of the methods.
SETTLE_S = 2.0      # Same as the tracking metrics.


def load(paths):
    """Pool the files. Return hidden (T, N, 128), alive (T, N), time_s (T,), mass_g (N,), onset_s."""
    files = [np.load(p) for p in paths]
    time_s = files[0]["time_s"]
    assert all(np.array_equal(f["time_s"], time_s) for f in files)
    keep = [f["mass_g"] <= MAX_MASS_G for f in files]
    hidden = np.concatenate([f["hidden"][:, k] for f, k in zip(files, keep)], axis=1)
    alive = np.concatenate([f["alive"][:, k] for f, k in zip(files, keep)], axis=1)
    mass = np.concatenate([f["mass_g"][k] for f, k in zip(files, keep)])
    return hidden, alive, time_s, mass, float(files[0]["onset_s"])


def samples(hidden, alive, mass, rows):
    """Flatten the chosen sample times to X (n, 128) and y (n,), robots that have not fallen only."""
    x = hidden[rows]
    ok = alive[rows]
    y = np.broadcast_to(mass, ok.shape)
    return x[ok].astype(np.float64), y[ok]


def fit(x, y):
    """Least squares with intercept. Return the weights (129,)."""
    a = np.hstack([x, np.ones((len(x), 1))])
    return np.linalg.lstsq(a, y, rcond=None)[0]


def r2(w, x, y):
    pred = x @ w[:-1] + w[-1]
    return 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fit", required=True)
    p.add_argument("--score", nargs="+", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    hf, af, time_s, mf, onset = load([args.fit])
    hs, as_, time_s2, ms, _ = load(args.score)
    assert np.array_equal(time_s, time_s2)
    late = np.flatnonzero(time_s >= onset + SETTLE_S)

    xf, yf = samples(hf, af, mf, late)
    xs, ys = samples(hs, as_, ms, late)
    w = fit(xf, yf)
    pred = xs @ w[:-1] + w[-1]
    result = {
        "name": args.name,
        "fit_samples": len(yf),
        "score_samples": len(ys),
        "r2_fit": r2(w, xf, yf),
        "r2_score": r2(w, xs, ys),
        "rmse_score_g": float(np.sqrt(((ys - pred) ** 2).mean())),
        # Exploratory: the same model at each sample time (before the payload appears, it must be near 0 or below).
        "r2_by_time": {f"{t:.1f}": round(float(r2(w, *samples(hs, as_, ms, [i]))), 4) for i, t in enumerate(time_s)},
    }
    print("%-10s fit n=%d R2=%.3f | score n=%d R2=%.3f, RMSE %.1f g" % (
        args.name, result["fit_samples"], result["r2_fit"], result["score_samples"], result["r2_score"], result["rmse_score_g"]))
    shown = [f"{t:.1f}" for t in time_s if abs(t % 1.0) < 1e-6 and 3.0 <= t <= 12.0]
    print("  R2 by time (s): " + "  ".join(f"{t}:{result['r2_by_time'][t]:.2f}" for t in shown))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
