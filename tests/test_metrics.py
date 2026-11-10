"""Tests for the evaluation metrics, with small hand-made arrays (no simulator)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from payload_duck import metrics  # noqa: E402

DT = 0.02


def make(T=10, N=3):
    vel = np.zeros((T, N, 3))
    cmd = np.zeros((N, 3))
    done = np.zeros((T, N), dtype=bool)
    fell = np.zeros((T, N), dtype=bool)
    return vel, cmd, done, fell


def test_first_true():
    mask = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=bool)
    assert metrics.first_true(mask).tolist() == [1, 2, -1]


def test_valid_mask_excludes_settle_and_end_step():
    done = np.zeros((6, 2), dtype=bool)
    done[3, 1] = True  # env 1 ends at step 3
    valid = metrics.valid_mask(done, dt=DT, settle_s=2 * DT)  # skip steps 0, 1
    assert valid[:, 0].tolist() == [False, False, True, True, True, True]
    assert valid[:, 1].tolist() == [False, False, True, False, False, False]


def test_perfect_tracking_gives_zero_error():
    vel, cmd, done, fell = make()
    cmd[:] = [0.3, 0.0, 0.5]
    vel[:] = cmd
    m = metrics.per_env(vel, cmd, done, fell, DT, settle_s=0.0)
    assert np.allclose(m["rmse_xy"], 0.0)
    assert np.allclose(m["rmse_yaw"], 0.0)
    assert np.allclose(m["mean_vx"], 0.3)
    assert not m["fell"].any()


def test_constant_error_rmse():
    vel, cmd, done, fell = make()
    cmd[:, 0] = 0.4
    vel[..., 0] = 0.1  # 0.3 too slow
    vel[..., 1] = 0.4  # 0.4 sideways drift -> |error| = 0.5
    m = metrics.per_env(vel, cmd, done, fell, DT, settle_s=0.0)
    assert np.allclose(m["rmse_xy"], 0.5)


def test_fall_detected_with_time():
    vel, cmd, done, fell = make(T=10, N=2)
    done[4, 0] = fell[4, 0] = True  # env 0 falls at step 4
    m = metrics.per_env(vel, cmd, done, fell, DT, settle_s=0.0)
    assert m["fell"][0] and not m["fell"][1]
    assert np.isclose(m["fall_time_s"][0], 5 * DT)
    assert np.isnan(m["fall_time_s"][1])
    assert m["n_valid_steps"].tolist() == [4, 10]


def test_fall_after_timeout_is_not_counted():
    # The env ends by time out first; a fall in its next episode does not count.
    vel, cmd, done, fell = make(T=10, N=1)
    done[5, 0] = True  # time out
    done[8, 0] = fell[8, 0] = True  # fall in the next episode
    m = metrics.per_env(vel, cmd, done, fell, DT, settle_s=0.0)
    assert not m["fell"][0]


def test_summarize_pools_over_valid_steps():
    vel, cmd, done, fell = make(T=10, N=2)
    vel[..., 0] = [1.0, 0.0]  # env 0 error 1, env 1 error 0
    done[2, 0] = fell[2, 0] = True  # env 0 has 2 valid steps, env 1 has 10
    m = metrics.per_env(vel, cmd, done, fell, DT, settle_s=0.0)
    s = metrics.summarize(m, np.arange(2))
    assert s["fall_rate"] == 0.5
    assert np.isclose(s["rmse_xy"], np.sqrt(2 / 12))
    assert np.isclose(s["median_fall_time_s"], 3 * DT)
