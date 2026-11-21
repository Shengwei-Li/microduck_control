"""Record the last hidden layer of the actor during the standard evaluation (for H4).

Run it from the upstream folder:
    cd ~/projectD/third_party/microduck_rl
    uv run --locked python <this repo>/scripts/probe_collect.py --task PayloadDuck-P3 \\
        --checkpoint logs/rsl_rl/velocity/<run>/model_3998.pt --seed 1000 \\
        --payload-mass-g 0 20 40 60 80 100 120 150 200 --out <file>.npz

The episodes are the same as in evaluate.py. Every --every steps, the script
stores the 128 outputs of the last hidden layer for each robot that has not
fallen. The file is large (about 70 MB), so keep it outside the repository.
probe_fit.py reads these files.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate  # noqa: E402
from payload_duck.eval_env import apply_commands, command_table, payload_per_env  # noqa: E402


def last_hidden(state, obs):
    """The actor by hand up to the last hidden layer: normalizer, then three layers with ELU."""
    x = (obs - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + 1e-2)
    for i in (0, 2, 4):
        x = torch.nn.functional.elu(x @ state[f"mlp.{i}.weight"].T + state[f"mlp.{i}.bias"])
    return x


def main():
    # Own option --every; all other options are those of evaluate.py (--out is the .npz file).
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--every", type=int, default=25, help="store one sample every N steps (25 = 0.5 s)")
    own, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = evaluate.parse_args()
    assert args.payloads is not None and args.payload_onset_s > 0.0

    torch.manual_seed(args.seed)
    env, venv, policy = evaluate.build(args)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = ckpt["actor_state_dict"]
    table, labels = command_table(args.envs_per_command, args.device, len(args.payloads))
    load_step, load_fn = evaluate.payload_loader(env, args)

    feats, steps, alive_at = [], [], []
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=args.device)
    worst = 0.0
    venv.reset()
    apply_commands(env, table)
    obs = venv.get_observations()
    with torch.inference_mode():
        for t in range(env.max_episode_length):
            if t == load_step:
                load_fn()
            action = policy(obs)
            if t % own.every == 0:
                h = last_hidden(state, obs["actor"])
                # Check: the hidden layer by hand gives the action of the policy.
                by_hand = h @ state["mlp.6.weight"].T + state["mlp.6.bias"]
                worst = max(worst, (by_hand - action).abs().max().item())
                feats.append(h.cpu().numpy().astype(np.float32))
                steps.append(t)
                alive_at.append(alive.cpu().numpy())
            obs, _, dones, _ = venv.step(action)
            apply_commands(env, table)
            alive &= ~dones.bool()
    assert worst < 1e-4, f"hidden layer by hand does not match the policy: {worst:.2e}"

    mass_kg, _ = payload_per_env(args.payloads, args.envs_per_command)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             hidden=np.stack(feats),                      # (samples in time, robots, 128)
             alive=np.stack(alive_at),                    # (samples in time, robots): not fallen before this step
             time_s=np.array(steps) * env.step_dt,        # (samples in time,)
             mass_g=np.array(mass_kg) * 1000.0,           # (robots,)
             command=np.array(labels),                    # (robots,)
             onset_s=args.payload_onset_s)
    print(f"largest action difference by hand: {worst:.2e}. hidden {np.stack(feats).shape}. Wrote {out}")
    env.close()


if __name__ == "__main__":
    main()
