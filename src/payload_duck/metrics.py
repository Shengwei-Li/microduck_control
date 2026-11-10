"""Evaluation metrics. Pure numpy, so that they can be tested without a simulator.

Shapes: T steps, N envs.
  vel     (T, N, 3)  measured body-frame velocity: lin x, lin y, yaw rate
  cmd     (N, 3)     fixed command of each env
  done    (T, N)     True at the step where the episode of the env ended
  fell    (T, N)     True at the step where the env ended by falling
"""

import numpy as np


def first_true(mask: np.ndarray) -> np.ndarray:
    """Index of the first True along axis 0, or -1 if there is none. Shape (N,)."""
    any_true = mask.any(axis=0)
    idx = mask.argmax(axis=0)
    return np.where(any_true, idx, -1)


def valid_mask(done: np.ndarray, dt: float, settle_s: float) -> np.ndarray:
    """Steps used for tracking metrics: after settle_s, before the episode ended.

    The step where the episode ends is excluded: the simulator already reset
    that env, so its measured state belongs to the next episode.
    """
    T, N = done.shape
    end = first_true(done)
    end = np.where(end < 0, T, end)  # No end: all steps are valid.
    t = np.arange(T)[:, None]
    return (t >= int(round(settle_s / dt))) & (t < end[None, :])


def per_env(vel, cmd, done, fell, dt, settle_s):
    """Metrics for each env. Returns a dict of arrays with shape (N,)."""
    valid = valid_mask(done, dt, settle_s)
    n_valid = valid.sum(axis=0)
    err = vel - cmd[None, :, :]
    err_xy_sq = (err[..., 0] ** 2 + err[..., 1] ** 2) * valid
    err_yaw_sq = (err[..., 2] ** 2) * valid
    safe_n = np.maximum(n_valid, 1)
    mean_vel = (vel * valid[..., None]).sum(axis=0) / safe_n[:, None]

    end = first_true(done)
    fell_idx = first_true(fell)
    # The env fell if its first episode end was a fall.
    has_fallen = (fell_idx >= 0) & (fell_idx == end)
    fall_time = np.where(has_fallen, (fell_idx + 1) * dt, np.nan)

    nan = np.full(n_valid.shape, np.nan)
    return {
        "fell": has_fallen,
        "fall_time_s": fall_time,
        "n_valid_steps": n_valid,
        "rmse_xy": np.where(n_valid > 0, np.sqrt(err_xy_sq.sum(axis=0) / safe_n), nan),
        "rmse_yaw": np.where(n_valid > 0, np.sqrt(err_yaw_sq.sum(axis=0) / safe_n), nan),
        "mean_vx": np.where(n_valid > 0, mean_vel[:, 0], nan),
        "mean_vy": np.where(n_valid > 0, mean_vel[:, 1], nan),
        "mean_wz": np.where(n_valid > 0, mean_vel[:, 2], nan),
    }


def summarize(env_metrics: dict, idx: np.ndarray) -> dict:
    """Summary over a group of envs (selected by the index array idx)."""
    m = {k: v[idx] for k, v in env_metrics.items()}
    n = m["n_valid_steps"]
    w = n / max(n.sum(), 1)  # Pool the squared errors over all valid steps.

    def pooled_rmse(key):
        x = np.nan_to_num(m[key])
        return float(np.sqrt((w * x**2).sum())) if n.sum() > 0 else float("nan")

    def weighted_mean(key):
        x = np.nan_to_num(m[key])
        return float((w * x).sum()) if n.sum() > 0 else float("nan")

    fall_times = m["fall_time_s"][m["fell"]]
    return {
        "n_envs": int(len(idx)),
        "fall_rate": float(m["fell"].mean()),
        "median_fall_time_s": float(np.median(fall_times)) if len(fall_times) else None,
        "rmse_xy": pooled_rmse("rmse_xy"),
        "rmse_yaw": pooled_rmse("rmse_yaw"),
        "mean_vx": weighted_mean("mean_vx"),
        "mean_vy": weighted_mean("mean_vy"),
        "mean_wz": weighted_mean("mean_wz"),
    }
