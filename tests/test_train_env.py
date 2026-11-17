"""Tests for the payload training env. The config tests do not start a simulation.
The event tests use a small fake env (no simulator)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mjlab_microduck")

from payload_duck import train_env  # noqa: E402


@pytest.fixture(scope="module")
def base_cfg():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    return load_env_cfg(train_env.BASE_TASK_ID)


def test_train_cfg_adds_only_the_payload(base_cfg):
    cfg = train_env.make_train_env_cfg(base_cfg, (0.0, 0.1))
    assert set(cfg.events) - set(base_cfg.events) == {"reset_payload", "pickup_payload"}
    assert cfg.events["reset_payload"].mode == "reset" and cfg.events["pickup_payload"].mode == "step"
    assert cfg.events["reset_payload"].params["mass_range_kg"] == (0.0, 0.1)
    # Training conditions stay: curriculum and pushes.
    assert set(cfg.curriculum) == set(base_cfg.curriculum) and "push_robot" in cfg.events
    assert cfg.scene.num_envs == base_cfg.scene.num_envs
    assert "reset_payload" not in base_cfg.events  # The base cfg is not changed.


def test_method_observations(base_cfg):
    def cfg_of(task):
        cfg = train_env.make_train_env_cfg(base_cfg, (0.0, 0.1))
        train_env.apply_method_obs(cfg, train_env.TRAIN_TASKS[task])
        return cfg.observations

    base = base_cfg.observations
    p1, p2, p3 = cfg_of("PayloadDuck-P1"), cfg_of("PayloadDuck-P2"), cfg_of("PayloadDuck-P3")
    # P1: the same observations as upstream.
    assert list(p1["actor"].terms) == list(base["actor"].terms) and p1["actor"].history_length is None
    # P2: the true mass is the last term for the actor and the critic. No history.
    assert list(p2["actor"].terms) == list(base["actor"].terms) + ["payload_mass"]
    assert list(p2["critic"].terms) == list(base["critic"].terms) + ["payload_mass"]
    assert p2["actor"].history_length is None
    # P3: the actor has a history and no mass. Only the critic has the mass.
    assert list(p3["actor"].terms) == list(base["actor"].terms)
    assert p3["actor"].history_length == train_env.HISTORY_STEPS and p3["critic"].history_length is None
    assert list(p3["critic"].terms)[-1] == "payload_mass"
    assert "payload_mass" not in base["critic"].terms  # The base cfg is not changed.


def test_tasks_register():
    from mjlab.tasks.registry import list_tasks, load_env_cfg

    train_env.register_train_tasks()
    train_env.register_train_tasks()  # A second call is harmless.
    assert set(train_env.TRAIN_TASKS) <= set(list_tasks())
    assert load_env_cfg("PayloadDuck-P0c").events["reset_payload"].params["mass_range_kg"] == (0.0, 0.0)


class FakeSim:
    """Records the calls that pickup_payload makes after a mass change."""

    def __init__(self, n):
        self.model = SimpleNamespace(body_mass=torch.zeros(n, 3), body_inertia=torch.zeros(n, 3, 3))
        self.calls = []

    def recompute_constants(self, level):
        self.calls.append("recompute")

    def forward(self):
        self.calls.append("forward")


def fake_env(n=1000):
    env = SimpleNamespace(num_envs=n, device="cpu", step_dt=0.02, common_step_counter=0,
                          episode_length_buf=torch.zeros(n, dtype=torch.long), sim=FakeSim(n))
    return env


def test_reset_then_pickup(monkeypatch):
    monkeypatch.setattr(train_env, "write_payload",
                        lambda env, ids, mass, mount: env.sim.model.body_mass.__setitem__((ids, 2), mass))
    torch.manual_seed(0)
    env = fake_env()
    train_env.reset_payload(env, None, (0.0, 0.1), (0.0, 10.0))
    assert env.payload_mass_kg.min() >= 0.0 and env.payload_mass_kg.max() <= 0.1
    assert 0.04 < env.payload_mass_kg.mean() < 0.06  # Uniform: mean near 0.05.
    assert env.sim.model.body_mass.sum() == 0.0 and not env.payload_loaded.any()

    # t = 5 s: about half of the envs have an onset time before 5 s.
    env.episode_length_buf[:] = 250
    env.common_step_counter = 7  # Not a multiple of PICKUP_EVERY_STEPS: nothing happens.
    train_env.pickup_payload(env, None)
    assert not env.payload_loaded.any() and env.sim.calls == []
    env.common_step_counter = 10
    train_env.pickup_payload(env, None)
    due = env.payload_onset_s <= 5.0
    assert (env.payload_loaded == due).all() and 400 < due.sum() < 600
    assert torch.equal(env.sim.model.body_mass[:, 2], env.payload_mass_kg * due)
    assert env.sim.calls == ["recompute", "forward"]

    # A reset of some envs removes their payload and draws a new one.
    ids = torch.arange(0, 100)
    old = env.payload_mass_kg.clone()
    train_env.reset_payload(env, ids, (0.0, 0.1), (0.0, 10.0))
    assert not env.payload_loaded[:100].any() and env.sim.model.body_mass[:100, 2].sum() == 0.0
    assert torch.equal(env.payload_mass_kg[100:], old[100:])
