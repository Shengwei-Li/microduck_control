"""Record what the runtime monitors need, during the standard evaluation (D4).

Run it from the upstream folder:
    cd ~/projectD/third_party/microduck_rl
    uv run --locked python <this repo>/scripts/monitor_collect.py [--task PayloadDuck-P3] \\
        --checkpoint logs/rsl_rl/velocity/<run>/model_3998.pt --seed 1000 \\
        --payload-mass-g 0 20 40 60 80 100 120 150 200 --probe <weights>.npy --out <file>.npz

The episodes are the same as in evaluate.py. For each control step from
--from-s to --to-s, the script stores for each robot:

    frame    the newest actor observation (61 values): what the robot measures
    action   the action of the policy (14 values)
    value    the output of the critic (it uses simulator state, see below)
    mass_est the payload mass from the linear probe of D3 (grams), if --probe is given

It also stores the step of the fall for each robot (-1 = no fall). The file
is large (about 270 MB), so keep it outside the repository.

The critic reads values that a real robot does not have (base velocity, foot
contact, and for P2 and P3 the true payload mass). The value signal is for
simulation only.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate  # noqa: E402
from payload_duck import train_env  # noqa: E402
from payload_duck.eval_env import apply_commands, command_table, payload_per_env  # noqa: E402
from probe_collect import last_hidden  # noqa: E402


def mlp(state, obs):
    """A full actor or critic by hand: last hidden layer, then the output layer."""
    return last_hidden(state, obs) @ state["mlp.6.weight"].T + state["mlp.6.bias"]


def newest_frame_columns(term_dims, steps):
    """Columns of the newest frame in an actor observation with history.

    term_dims: size of each term for one frame. With history, the observation
    holds each term as `steps` frames, oldest first. steps = 0: no history.
    """
    if steps == 0:
        return list(range(sum(term_dims)))
    cols, start = [], 0
    for d in term_dims:
        cols += list(range(start + d * (steps - 1), start + d * steps))
        start += d * steps
    return cols


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--from-s", type=float, default=3.0)
    extra.add_argument("--to-s", type=float, default=13.0)
    extra.add_argument("--probe", default=None, help="weights from probe_fit.py --save-weights")
    own, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = evaluate.parse_args()
    assert args.payloads is not None and args.payload_onset_s > 0.0

    torch.manual_seed(args.seed)
    env, venv, policy = evaluate.build(args)
    dev = args.device
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    actor, critic = ckpt["actor_state_dict"], ckpt["critic_state_dict"]
    probe = None if own.probe is None else torch.tensor(np.load(own.probe), dtype=torch.float32, device=dev)

    manager = env.observation_manager
    steps = train_env.TRAIN_TASKS[args.task].actor_history if args.task else 0
    names = list(manager.active_terms["actor"])
    dims = [d[0] // max(steps, 1) for d in manager.group_obs_term_dim["actor"]]
    if names[-1] == "payload_mass":  # P2: the mass input is not a measurement of the robot.
        names, dims = names[:-1], dims[:-1]
    cols = torch.tensor(newest_frame_columns(dims, steps), device=dev)

    table, labels = command_table(args.envs_per_command, dev, len(args.payloads))
    load_step, load_fn = evaluate.payload_loader(env, args)
    terms = env.termination_manager
    first, last = round(own.from_s / env.step_dt), round(own.to_s / env.step_dt)

    n = env.num_envs
    frame, action_log, value, mass_est = [], [], [], []
    fall_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    done_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    worst = 0.0
    venv.reset()
    apply_commands(env, table)
    obs = venv.get_observations()
    with torch.inference_mode():
        for t in range(env.max_episode_length):
            if t == load_step:
                load_fn()
            action = policy(obs)
            if first <= t < last:
                h = last_hidden(actor, obs["actor"])
                worst = max(worst, (h @ actor["mlp.6.weight"].T + actor["mlp.6.bias"] - action).abs().max().item())
                frame.append(obs["actor"][:, cols].half().cpu())
                action_log.append(action.half().cpu())
                value.append(mlp(critic, obs["critic"])[:, 0].float().cpu())
                if probe is not None:
                    mass_est.append((h @ probe[:-1] + probe[-1]).cpu())
            obs, _, dones, _ = venv.step(action)
            apply_commands(env, table)
            running = done_step < 0
            fall_step[running & terms.get_term("fell_over")] = t
            done_step[running & dones.bool()] = t
    assert worst < 1e-4, f"actor by hand does not match the policy: {worst:.2e}"

    mass_kg, _ = payload_per_env(args.payloads, args.envs_per_command)
    robot = env.command_manager.get_term("twist").robot
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             frame=torch.stack(frame).numpy(),            # (steps, robots, 61) float16
             action=torch.stack(action_log).numpy(),      # (steps, robots, 14) float16
             value=torch.stack(value).numpy(),            # (steps, robots)
             mass_est=torch.stack(mass_est).numpy() if mass_est else np.zeros(0),
             fall_step=fall_step.cpu().numpy(),           # (robots,) step of the fall in the episode, -1 = none
             done_step=done_step.cpu().numpy(),           # (robots,) step of the first end of episode, -1 = none
             first_step=first, step_dt=env.step_dt, onset_s=args.payload_onset_s,
             mass_g=np.array(mass_kg) * 1000.0, command=np.array(labels),
             term_names=np.array(names), term_dims=np.array(dims), joint_names=np.array(robot.joint_names))
    print(f"actor by hand: largest difference {worst:.2e}. frame {tuple(torch.stack(frame).shape)}. "
          f"falls {int((fall_step >= 0).sum())} of {n}. Wrote {out}")
    env.close()


if __name__ == "__main__":
    main()
