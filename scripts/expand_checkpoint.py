"""Make the start checkpoint of a method with a larger network input (P2, P3).

Run it from the upstream folder:
    cd ~/projectD/third_party/microduck_rl
    uv run --locked python <this repo>/scripts/expand_checkpoint.py \\
        --task PayloadDuck-P3 --checkpoint logs/rsl_rl/velocity/<P0 run>/model_2999.pt \\
        --out logs/rsl_rl/velocity/<P0 run>_PayloadDuck-P3/model_2999.pt

It builds a small env of the task to read the observation layout. Then it
changes the checkpoint (see src/payload_duck/checkpoint.py). Last, it runs the
new actor in the env for 60 steps and compares each action with the action of
the P0 actor: the two must be the same.
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from payload_duck import checkpoint, train_env  # noqa: E402


def group_layout(names, dims, has_mass, steps):
    """Layout of one observation group.

    names, dims: the terms of the group and their sizes (with history: size x steps).
    has_mass: the last term is the new payload mass. steps: history length, 0 = none.
    """
    if has_mass:
        assert names[-1] == "payload_mass" and dims[-1] == 1
        dims = dims[:-1]
    if steps > 0:
        layout = checkpoint.history([d // steps for d in dims], steps)
    else:
        layout = checkpoint.appended(sum(dims), 0)
    return checkpoint.then(layout, 1 if has_mass else 0)


def p0_actor(state, obs_old):
    """The P0 actor by hand: normalizer, three hidden layers (ELU), output layer."""
    x = (obs_old - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + 1e-2)
    for i in (0, 2, 4):
        x = torch.nn.functional.elu(x @ state[f"mlp.{i}.weight"].T + state[f"mlp.{i}.bias"])
    return x @ state["mlp.6.weight"].T + state["mlp.6.bias"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(train_env.TRAIN_TASKS))
    p.add_argument("--checkpoint", required=True, help="P0 checkpoint (model_N.pt)")
    p.add_argument("--out", required=True, help="new checkpoint file")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    train_env.register_train_tasks()
    method = train_env.TRAIN_TASKS[args.task]
    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = 16
    agent_cfg = load_rl_cfg(args.task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    manager = env.observation_manager
    layouts = {}
    for group in ("actor", "critic"):
        dims = [d[0] for d in manager.group_obs_term_dim[group]]
        steps = method.actor_history if group == "actor" else 0
        layouts[group] = group_layout(manager.active_terms[group], dims, group in method.mass_obs, steps)

    old = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    new = checkpoint.expand_checkpoint(old, layouts["actor"], layouts["critic"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, out)
    for key in ("actor_state_dict", "critic_state_dict"):
        print(f"{key}: input {old[key]['mlp.0.weight'].shape[1]} -> {new[key]['mlp.0.weight'].shape[1]}")

    # Check in the env. The runner load is strict: all tensor sizes must fit.
    runner = load_runner_cls(args.task)(venv, asdict(agent_cfg), device=args.device)
    runner.load(str(out), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    state = {k: v.to(args.device) for k, v in old["actor_state_dict"].items()}
    new_cols, old_cols = (c.to(args.device) for c in checkpoint.live_columns(layouts["actor"]))

    obs = venv.get_observations()
    worst = 0.0
    with torch.inference_mode():
        for _ in range(60):
            action = policy(obs)
            obs_old = torch.zeros(env.num_envs, len(old_cols), device=args.device)
            obs_old[:, old_cols] = obs["actor"][:, new_cols]  # What the P0 actor would see.
            worst = max(worst, (action - p0_actor(state, obs_old)).abs().max().item())
            obs, _, _, _ = venv.step(action)
    print(f"largest action difference to P0 in 60 steps: {worst:.2e}")
    assert worst < 1e-4, "the new actor does not start with the behaviour of P0"
    print(f"wrote {out}")
    env.close()


if __name__ == "__main__":
    main()
