"""Runtime monitors: signals, accumulation, and thresholds (D4).

A monitor reads one number per robot and control step (a signal). A signal is
high when the robot is outside its normal conditions. "Normal" here means: the
payload is inside the training range of the policy (0 to 100 g).

All signals use the same steps, so that differences between monitors come from
the signal:

1. Standardize: z = (signal - mu) / sigma. mu and sigma are the median and the
   scaled MAD of normal data (fit).
2. Accumulate with a CUSUM: g = max(0, g + z - kappa). With kappa = None there
   is no accumulation and the score is z (a plain threshold).
3. Threshold: split-conformal quantile of the largest score of each normal
   calibration episode, for a target false-alarm rate per episode.

Arrays have the shape (steps, robots). A step that does not count (after the
end of the episode) is NaN in the signal: the CUSUM keeps its value there.

The CUSUM and the threshold are the same as in the earlier arm project
(lac.monitor), without the logarithm: these signals can be negative.
"""

import math
from dataclasses import dataclass

import numpy as np

NECK_JOINT = "neck_pitch"


def term_slices(term_names, term_dims):
    """Position of each observation term in the frame: name -> slice."""
    out, start = {}, 0
    for name, dim in zip(term_names, term_dims):
        out[str(name)] = slice(start, start + int(dim))
        start += int(dim)
    return out


def neck_error(frame, terms, joint_names):
    """Position error of the neck joint (rad): target minus measured position.

    The frame holds the last action and the joint positions. Both are relative
    to the default pose and the action scale is 1, so the error is their difference.
    """
    j = [str(n) for n in joint_names].index(NECK_JOINT)
    return frame[..., terms["actions"].start + j] - frame[..., terms["joint_pos"].start + j]


@dataclass
class Standardizer:
    mu: float
    sigma: float

    @classmethod
    def fit(cls, normal_signal):
        x = np.asarray(normal_signal, dtype=float)
        x = x[np.isfinite(x)]
        mu = float(np.median(x))
        mad = float(np.median(np.abs(x - mu)))
        return cls(mu, 1.4826 * mad if mad > 0 else float(x.std() or 1.0))

    def __call__(self, signal):
        return (np.asarray(signal, dtype=float) - self.mu) / self.sigma


def cusum(z, kappa):
    """Scores (steps, robots) from the standardized signal. kappa None: the score is z."""
    z = np.asarray(z, dtype=float)
    if kappa is None:
        return z.copy()
    out = np.full(z.shape, np.nan)
    g = np.zeros(z.shape[1])
    for t in range(z.shape[0]):
        ok = np.isfinite(z[t])
        g = np.where(ok, np.maximum(0.0, g + np.nan_to_num(z[t]) - kappa), g)
        out[t] = np.where(ok, g, np.nan)
    return out


def episode_max(scores):
    """Largest finite score of each robot (steps, robots) -> (robots,). -inf if there is none."""
    s = np.where(np.isfinite(scores), scores, -np.inf)
    return s.max(axis=0)


def conformal_threshold(normal_episode_max, alpha):
    """Split-conformal threshold: alarm when score > threshold. +inf if there are too few episodes."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    s = np.sort(np.asarray(normal_episode_max, dtype=float))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float(s[k - 1]) if 1 <= k <= len(s) else float("inf")


def first_alarm(scores, threshold):
    """Step of the first score above the threshold for each robot; -1 if there is none."""
    hit = np.isfinite(scores) & (scores > threshold)
    return np.where(hit.any(axis=0), hit.argmax(axis=0), -1)
