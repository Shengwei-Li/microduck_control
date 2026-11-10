"""Tests for the evaluation env config. They need the upstream environment
(mjlab, mjlab_microduck) but do not start a simulation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from payload_duck import eval_env  # noqa: E402

pytest.importorskip("mjlab_microduck")


@pytest.fixture(scope="module")
def base_cfg():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    return load_env_cfg(eval_env.TASK_ID)


def test_eval_cfg_fixed_conditions(base_cfg):
    cfg = eval_env.make_eval_env_cfg(base_cfg, envs_per_command=4, seed=1000)
    assert cfg.scene.num_envs == 4 * len(eval_env.EVAL_COMMANDS)
    assert cfg.seed == 1000
    assert cfg.curriculum == {}
    assert "push_robot" not in cfg.events
    assert cfg.events["randomize_com"].params["ranges"] == (-0.015, 0.015)
    twist = cfg.commands["twist"]
    assert twist.rel_standing_envs == 0.0
    assert twist.rel_turn_in_place_envs == 0.0
    assert twist.resampling_time_range[0] >= 1e6
    for name in ("head_pose", "body_pose"):
        assert all(r == (0.0, 0.0) for r in cfg.commands[name].ranges)


def test_base_cfg_not_changed(base_cfg):
    eval_env.make_eval_env_cfg(base_cfg, envs_per_command=4, seed=1000)
    assert "push_robot" in base_cfg.events
    assert len(base_cfg.curriculum) > 0


def test_eval_cfg_with_payloads(base_cfg):
    payloads = [(0.0, 0.0), (0.1, 0.02), (0.2, -0.02)]
    cfg = eval_env.make_eval_env_cfg(base_cfg, envs_per_command=2, seed=1000, payloads=payloads)
    block = 2 * len(eval_env.EVAL_COMMANDS)
    assert cfg.scene.num_envs == 3 * block
    params = cfg.events["set_payload"].params
    assert params["mass_kg"][0] == 0.0 and params["mass_kg"][block] == 0.1 and params["mass_kg"][-1] == 0.2
    assert params["offset_m"][block] == [0.02, 0.0, 0.0]
    # The base cfg keeps the upstream robot and has no payload event.
    assert "set_payload" not in base_cfg.events
    assert cfg.scene.entities["robot"].spec_fn is not base_cfg.scene.entities["robot"].spec_fn


def test_eval_commands_inside_training_range():
    for vx, vy, wz in eval_env.EVAL_COMMANDS.values():
        assert -0.4 <= vx <= 0.4 and -0.3 <= vy <= 0.3 and -1.0 <= wz <= 1.0


def test_command_table_blocks():
    table, labels = eval_env.command_table(envs_per_command=3, device="cpu")
    names = list(eval_env.EVAL_COMMANDS)
    assert table.shape == (3 * len(names), 3)
    assert labels[:3] == [names[0]] * 3 and labels[3] == names[1]
    assert tuple(table[3].tolist()) == pytest.approx(eval_env.EVAL_COMMANDS[names[1]])
    # Two conditions: the same block of commands two times.
    table2, labels2 = eval_env.command_table(envs_per_command=3, device="cpu", n_conditions=2)
    assert table2.shape == (2 * 3 * len(names), 3)
    assert labels2 == labels + labels
    assert (table2[: len(labels)] == table).all() and (table2[len(labels):] == table).all()
