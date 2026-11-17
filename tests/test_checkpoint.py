"""Tests for the checkpoint change of P2 and P3. Only torch: no simulator."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from payload_duck import checkpoint  # noqa: E402


def model_state(in_dim, out_dim=4, hidden=8):
    """A small state dict with the keys of an rsl_rl MLPModel."""
    return {
        "obs_normalizer._mean": torch.randn(1, in_dim),
        "obs_normalizer._var": torch.rand(1, in_dim) + 0.5,
        "obs_normalizer._std": torch.rand(1, in_dim) + 0.5,
        "obs_normalizer.count": torch.tensor(1000),
        "mlp.0.weight": torch.randn(hidden, in_dim),
        "mlp.0.bias": torch.randn(hidden),
        "mlp.2.weight": torch.randn(out_dim, hidden),
        "mlp.2.bias": torch.randn(out_dim),
    }


def forward(state, obs):
    """Normalizer, one hidden layer (ELU), output layer."""
    x = (obs - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + 1e-2)
    x = torch.nn.functional.elu(x @ state["mlp.0.weight"].T + state["mlp.0.bias"])
    return x @ state["mlp.2.weight"].T + state["mlp.2.bias"]


def test_layouts():
    assert checkpoint.appended(3, 1) == ([0, 1, 2, -1], [True, True, True, False])
    # Terms of size 1 and 2, with 2 frames: A_old, A_new, B_old (2), B_new (2).
    source, live = checkpoint.history([1, 2], 2)
    assert source == [0, 0, 1, 2, 1, 2]
    assert live == [False, True, False, False, True, True]
    assert checkpoint.then((source, live), 1) == (source + [-1], live + [False])


def test_appended_input_does_not_change_the_output():
    torch.manual_seed(0)
    old = model_state(5)
    new = checkpoint.expand_model(old, checkpoint.appended(5, 1))
    obs = torch.randn(10, 5)
    mass = torch.randn(10, 1) * 100.0  # Any value: its weights are 0.
    assert torch.allclose(forward(new, torch.cat([obs, mass], dim=1)), forward(old, obs), atol=1e-6)
    assert new["obs_normalizer._mean"][0, 5] == 0.0 and new["obs_normalizer._std"][0, 5] == 1.0
    assert new["mlp.2.weight"] is old["mlp.2.weight"]  # The other layers are the same.


def test_history_uses_only_the_newest_frame():
    torch.manual_seed(0)
    dims, steps = [2, 3], 4
    old = model_state(sum(dims))
    layout = checkpoint.history(dims, steps)
    new = checkpoint.expand_model(old, layout)
    # Build a history observation in the mjlab order: for each term, frames oldest to newest.
    frames = torch.randn(steps, 10, sum(dims))  # frames[-1] is the newest.
    a = frames[:, :, :2].permute(1, 0, 2).reshape(10, -1)
    b = frames[:, :, 2:].permute(1, 0, 2).reshape(10, -1)
    obs_new = torch.cat([a, b], dim=1)
    assert obs_new.shape[1] == len(layout[0]) == sum(dims) * steps
    assert torch.allclose(forward(new, obs_new), forward(old, frames[-1]), atol=1e-6)
    # All frames of an input have the statistics of that input.
    assert torch.equal(new["obs_normalizer._mean"][0, :8:2], old["obs_normalizer._mean"][0, 0].repeat(4))


def test_checkpoint_keeps_the_optimizer_in_step():
    torch.manual_seed(0)
    actor, critic = model_state(5), model_state(7, out_dim=1)
    slot = lambda w: {"step": torch.tensor(9.0), "exp_avg": torch.randn_like(w), "exp_avg_sq": torch.rand_like(w)}
    params = [actor["mlp.0.weight"], actor["mlp.0.bias"], critic["mlp.0.weight"], critic["mlp.0.bias"]]
    ckpt = {
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "optimizer_state_dict": {"state": {i: slot(w) for i, w in enumerate(params)}, "param_groups": [{"lr": 1e-3}]},
        "iter": 2999,
        "infos": {"env_state": {"common_step_counter": 123}},
    }
    new = checkpoint.expand_checkpoint(ckpt, checkpoint.history([2, 3], 3), checkpoint.appended(7, 1))
    assert new["actor_state_dict"]["mlp.0.weight"].shape == (8, 15)
    assert new["critic_state_dict"]["mlp.0.weight"].shape == (8, 8)
    state = new["optimizer_state_dict"]["state"]
    assert state[0]["exp_avg"].shape == (8, 15) and state[2]["exp_avg_sq"].shape == (8, 8)
    assert state[1]["exp_avg"].shape == (8,) and state[0]["step"] == 9.0
    # New columns start at 0. Old columns keep their values.
    assert torch.equal(state[2]["exp_avg"][:, :7], ckpt["optimizer_state_dict"]["state"][2]["exp_avg"])
    assert state[2]["exp_avg"][:, 7].abs().sum() == 0.0
    # History (actor): the mean of the squared gradient of each frame is that of its input.
    old_sq = ckpt["optimizer_state_dict"]["state"][0]["exp_avg_sq"]
    assert torch.equal(state[0]["exp_avg_sq"][:, 0:6:2], old_sq[:, 0:1].expand(8, 3))  # Input 0, frames 0..2.
    assert state[0]["exp_avg"][:, 0:4].abs().sum() == 0.0  # Old frames of term A: no gradient mean yet.
    assert torch.equal(state[0]["exp_avg"][:, 4:6], ckpt["optimizer_state_dict"]["state"][0]["exp_avg"][:, 0:2])
    assert new["iter"] == 2999 and new["infos"] == ckpt["infos"]
    assert ckpt["actor_state_dict"]["mlp.0.weight"].shape == (8, 5)  # The input is not changed.
