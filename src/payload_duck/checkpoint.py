"""Change the P0 checkpoint to a network with a larger input (methods P2, P3).

P2 and P3 give the networks more inputs than P0. We want to continue the
training of P0, so the new network must start with the behaviour of P0:

  - Each old input goes to one column of the new input (its "live" column).
    The weights of the first layer move with it.
  - All other new columns get weight 0. They do not change the output at the
    start. The training can then learn to use them.

Only the first layer and the input normalizer change. The rest of the network
is the same.

A layout is a pair (source, live) with one entry for each new column:
  source  the index of the old input that the column shows, or -1 for an
          input that P0 did not have;
  live    True if the column gets the P0 weights.

  appended(61, 1)          old inputs 0..60, then one new input
  history([3, 14], 25)     each term has 25 frames, oldest first (mjlab order:
                           term A frames 0..24, then term B frames 0..24)

In a history, all frames of an input share the normalizer statistics of that
input, but only the newest frame is live.
"""

import torch

Layout = tuple[list[int], list[bool]]


def appended(old_dim: int, extra: int) -> Layout:
    """Layout: the old inputs, then `extra` new inputs."""
    return list(range(old_dim)) + [-1] * extra, [True] * old_dim + [False] * extra


def history(term_dims: list[int], steps: int) -> Layout:
    """Layout: each term (sizes term_dims) has `steps` frames, oldest first."""
    source, live, start = [], [], 0
    for dim in term_dims:
        for frame in range(steps):
            source += list(range(start, start + dim))
            live += [frame == steps - 1] * dim  # Only the newest frame.
        start += dim
    return source, live


def then(layout: Layout, extra: int) -> Layout:
    """Add `extra` new inputs after a layout."""
    return layout[0] + [-1] * extra, layout[1] + [False] * extra


def live_columns(layout: Layout) -> tuple[torch.Tensor, torch.Tensor]:
    """Index tensors (new columns that are live, the old inputs that they show)."""
    source, live = layout
    new = [c for c, is_live in enumerate(live) if is_live]
    old = [source[c] for c in new]
    assert sorted(old) == list(range(len(old))), "each old input needs exactly one live column"
    return torch.tensor(new), torch.tensor(old)


def expand_weight(weight: torch.Tensor, layout: Layout) -> torch.Tensor:
    """First-layer tensor (out, old_dim) -> (out, new_dim). Columns that are not live are 0."""
    new, old = live_columns(layout)
    out = torch.zeros(weight.shape[0], len(layout[0]), dtype=weight.dtype, device=weight.device)
    out[:, new] = weight[:, old]
    return out


def expand_stat(stat: torch.Tensor, layout: Layout, default: float) -> torch.Tensor:
    """Tensor (rows, old_dim) -> (rows, new_dim): each column gets the values of its source.

    Inputs without a source get `default`. Used for the normalizer (rows = 1).
    """
    source = torch.tensor(layout[0])
    out = torch.full((stat.shape[0], len(source)), default, dtype=stat.dtype, device=stat.device)
    out[:, source >= 0] = stat[:, source[source >= 0]]
    return out


def expand_model(state: dict, layout: Layout) -> dict:
    """State dict of an rsl_rl MLPModel (actor or critic) with the new input layout."""
    out = dict(state)
    out["mlp.0.weight"] = expand_weight(state["mlp.0.weight"], layout)
    # A new input starts with mean 0 and standard deviation 1.
    out["obs_normalizer._mean"] = expand_stat(state["obs_normalizer._mean"], layout, 0.0)
    out["obs_normalizer._var"] = expand_stat(state["obs_normalizer._var"], layout, 1.0)
    out["obs_normalizer._std"] = expand_stat(state["obs_normalizer._std"], layout, 1.0)
    return out


def expand_checkpoint(ckpt: dict, actor_layout: Layout, critic_layout: Layout) -> dict:
    """Checkpoint of the upstream runner with new input layouts for the actor and the critic.

    The optimizer (Adam) keeps two running means for each weight: of the
    gradient (exp_avg) and of its square (exp_avg_sq). Adam divides each step
    by the root of the second one. These tensors change with the weights:

      exp_avg      new columns start at 0, like the weights.
      exp_avg_sq   the frames of a history get the value of their input (the
                   live column). With 0 here, the first steps of the new
                   weights would be very large. In a history there are many
                   new weights, and these steps together destroy the policy.
    """
    out = dict(ckpt)
    by_shape = {}  # Shape of the old first-layer weight -> layout.
    for key, layout in (("actor_state_dict", actor_layout), ("critic_state_dict", critic_layout)):
        by_shape[tuple(ckpt[key]["mlp.0.weight"].shape)] = layout
        out[key] = expand_model(ckpt[key], layout)
    assert len(by_shape) == 2, "the actor and the critic must have different input sizes"

    optimizer = dict(ckpt["optimizer_state_dict"])
    optimizer["state"] = {}
    for index, slot in ckpt["optimizer_state_dict"]["state"].items():
        slot = dict(slot)
        for name, value in slot.items():
            layout = by_shape.get(tuple(value.shape)) if torch.is_tensor(value) else None
            if layout is None:
                continue
            if name == "exp_avg_sq":
                slot[name] = expand_stat(value, layout, 0.0)
            else:
                slot[name] = expand_weight(value, layout)
        optimizer["state"][index] = slot
    out["optimizer_state_dict"] = optimizer
    return out
