"""Tests of the monitor library (no simulator needed)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from payload_duck import monitor  # noqa: E402


def test_neck_error_reads_the_right_columns():
    terms = monitor.term_slices(["base_ang_vel", "joint_pos", "actions"], [3, 14, 14])
    joints = ["a"] * 5 + ["neck_pitch"] + ["b"] * 8
    frame = np.zeros((2, 4, 31))
    frame[..., 3 + 5] = 0.2    # measured neck position
    frame[..., 17 + 5] = 0.5   # neck target (last action)
    assert np.allclose(monitor.neck_error(frame, terms, joints), 0.3)


def test_standardizer_uses_median_and_mad():
    s = monitor.Standardizer.fit(np.array([1.0, 2.0, 3.0, 4.0, 100.0, np.nan]))
    assert s.mu == 3.0
    assert np.isclose(s.sigma, 1.4826 * 1.0)
    assert np.isclose(s(3.0 + s.sigma), 1.0)


def test_cusum_by_hand():
    z = np.array([[1.0], [1.0], [-5.0], [2.0], [np.nan], [1.0]])
    g = monitor.cusum(z, kappa=0.5)[:, 0]
    assert np.allclose(g[:4], [0.5, 1.0, 0.0, 1.5])
    assert np.isnan(g[4])           # A step that does not count gives no score ...
    assert np.isclose(g[5], 2.0)    # ... and the sum continues after it.


def test_no_accumulation_gives_the_signal():
    z = np.array([[0.3, -1.0], [2.0, 0.1]])
    assert np.array_equal(monitor.cusum(z, None), z)


def test_conformal_threshold_and_false_alarm_rate():
    rng = np.random.default_rng(0)
    cal = rng.normal(size=1999)
    thr = monitor.conformal_threshold(cal, alpha=0.05)
    assert thr == np.sort(cal)[1899]  # ceil(2000 * 0.95) = 1900 -> the 1900th smallest value
    new = rng.normal(size=20000)
    assert abs((new > thr).mean() - 0.05) < 0.01
    assert monitor.conformal_threshold(np.arange(5.0), alpha=0.05) == float("inf")


def test_first_alarm_and_episode_max():
    scores = np.array([[0.0, 0.0, 0.0], [2.0, 0.5, np.nan], [3.0, 0.7, np.nan]])
    assert list(monitor.first_alarm(scores, 1.0)) == [1, -1, -1]
    assert list(monitor.episode_max(scores)) == [3.0, 0.7, 0.0]
