"""Evaluate a walking policy with fixed commands and fixed conditions.

Run it from the upstream folder, so that it uses the upstream environment:
    cd ~/projectD/third_party/microduck_rl
    uv run --locked python /mnt/d/EE_study/mphil_ready/projectD_rl_payload_biped/scripts/evaluate.py \
        --checkpoint logs/rsl_rl/velocity/<run>/model_2999.pt --seed 1000 --out <folder>

Each env runs one episode (20 s) with one fixed command (see eval_env.py).
With --payload-mass-g, the robot holds a payload in its beak. The payload
appears at --payload-onset-s (default 5 s): the robot "picks up" the object.
Writes <out>/summary.json (per command and overall) and <out>/per_env.csv.

Seeds: training uses 0-99. Evaluation uses 1000-1999 for development
(choosing settings) and 2000-2999 for the final tests.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from payload_duck import metrics  # noqa: E402
from payload_duck.eval_env import (  # noqa: E402
    EVAL_COMMANDS,
    TASK_ID,
    apply_commands,
    command_table,
    make_eval_env_cfg,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="model_N.pt file")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--envs-per-command", type=int, default=128)
    p.add_argument("--settle-s", type=float, default=2.0, help="ignore the first seconds for tracking metrics")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--payload-mass-g", type=float, nargs="+", default=None,
                   help="payload masses in grams; without it the robot has no payload body")
    p.add_argument("--payload-dx-mm", type=float, nargs="+", default=[0.0],
                   help="payload forward offsets in mm (+ is forward); all mass x offset pairs are tested")
    p.add_argument("--payload-mount", choices=["mouth", "trunk"], default="mouth")
    p.add_argument("--payload-onset-s", type=float, default=5.0,
                   help="time when the payload appears; 0 = from the start of the episode")
    args = p.parse_args()
    args.payloads = payload_conditions(args.payload_mass_g, args.payload_dx_mm)
    return args


def payload_conditions(mass_g, dx_mm):
    """All (mass_kg, dx_m) pairs, or None if no mass is given."""
    if mass_g is None:
        return None
    return [(m / 1000.0, dx / 1000.0) for m in mass_g for dx in dx_mm]


def build(args, viewer=None):
    """Build the evaluation env and load the policy (same steps as upstream `play`).

    viewer: dict of ViewerConfig fields (width, height, distance, ...). If given,
    offscreen rendering is on and env.render() gives frames.
    """
    import mjlab_microduck.tasks  # noqa: F401  Registers the Microduck tasks.
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = make_eval_env_cfg(load_env_cfg(TASK_ID), args.envs_per_command, args.seed,
                                payloads=args.payloads, mount=args.payload_mount,
                                loaded_at_start=args.payload_onset_s <= 0.0)
    render_mode = None
    if viewer is not None:
        for key, value in viewer.items():
            setattr(env_cfg.viewer, key, value)
        render_mode = "rgb_array"
    agent_cfg = load_rl_cfg(TASK_ID)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=render_mode)
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(venv, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    return env, venv, policy


def payload_loader(env, args):
    """Return (step, function) for mid-episode loading, or (None, None) if not used."""
    if args.payloads is None or args.payload_onset_s <= 0.0:
        return None, None
    from payload_duck import payload
    from payload_duck.eval_env import payload_per_env

    mass, offset = payload_per_env(args.payloads, args.envs_per_command)
    step = round(args.payload_onset_s / env.step_dt)
    return step, lambda: payload.load(env, mass, offset, args.payload_mount)


def rollout(env, venv, policy, table, load_step=None, load_fn=None):
    """Run one episode length. Return numpy arrays vel (T,N,3), done, fell, nan (T,N).

    If load_step is given, load_fn() runs before that step (the payload appears).
    """
    robot = env.command_manager.get_term("twist").robot
    terms = env.termination_manager
    T = env.max_episode_length
    vel, done, fell, nan = [], [], [], []

    venv.reset()
    apply_commands(env, table)
    obs = venv.get_observations()  # Recompute: the reset drew random commands.
    with torch.inference_mode():
        for t in range(T):
            if t == load_step:
                load_fn()
            obs, _, dones, _ = venv.step(policy(obs))
            # An env that ended was reset and got a random command. It no longer
            # counts, but give it the fixed command again to keep the table simple.
            apply_commands(env, table)
            v = torch.cat([robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]], dim=1)
            vel.append(v.cpu())
            done.append(dones.bool().cpu())
            fell.append(terms.get_term("fell_over").cpu())
            nan.append(terms.get_term("nan_state").cpu())
    return (torch.stack(vel).numpy(), torch.stack(done).numpy(),
            torch.stack(fell).numpy(), torch.stack(nan).numpy())


def upstream_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()
    env, venv, policy = build(args)
    # One condition = one payload (mass, offset). Without payload there is one condition.
    conditions = args.payloads if args.payloads is not None else [(0.0, 0.0)]
    table, labels = command_table(args.envs_per_command, args.device, len(conditions))
    load_step, load_fn = payload_loader(env, args)
    vel, done, fell, nan = rollout(env, venv, policy, table, load_step, load_fn)
    dt = env.step_dt

    # Tracking metrics start settle_s after the payload appears (or after the start).
    onset_s = args.payload_onset_s if load_step is not None else 0.0
    m = metrics.per_env(vel, table.cpu().numpy(), done, fell, dt, onset_s + args.settle_s)
    labels_arr = np.array(labels)
    block = args.envs_per_command * len(EVAL_COMMANDS)
    cond_of_env = np.arange(len(labels)) // block
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "upstream_commit": upstream_commit(),
        "seed": args.seed,
        "envs_per_command": args.envs_per_command,
        "settle_s": args.settle_s,
        "episode_s": env.max_episode_length * dt,
        "payload_body": args.payloads is not None,
        "payload_mount": args.payload_mount if args.payloads is not None else None,
        "payload_onset_s": onset_s,  # Fall times count from the start of the episode.
        "nan_episodes": int((metrics.first_true(nan) >= 0).sum()),
        "conditions": [],
    }
    for k, (mass_kg, dx_m) in enumerate(conditions):
        in_cond = cond_of_env == k
        cond = {
            "payload_mass_g": round(mass_kg * 1000.0, 3),
            "payload_dx_mm": round(dx_m * 1000.0, 3),
            "overall": metrics.summarize(m, np.flatnonzero(in_cond)),
            "commands": {},
        }
        for name, value in EVAL_COMMANDS.items():
            s = metrics.summarize(m, np.flatnonzero(in_cond & (labels_arr == name)))
            cond["commands"][name] = {"command": value, **s}
        summary["conditions"].append(cond)
    summary["wall_time_s"] = round(time.time() - t0, 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "per_env.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = list(m)
        w.writerow(["env", "payload_mass_g", "payload_dx_mm", "command"] + keys)
        for i, label in enumerate(labels):
            c = summary["conditions"][cond_of_env[i]]
            w.writerow([i, c["payload_mass_g"], c["payload_dx_mm"], label] + [m[k][i] for k in keys])

    # Short table on the screen: one line per condition and command.
    print(f"{'mass_g':>6s} {'dx_mm':>6s} {'command':13s} {'fall':>6s} {'t_fall':>6s} {'rmse_xy':>8s} {'rmse_yaw':>8s} "
          f"{'vx':>6s} {'vy':>6s} {'wz':>6s}")
    for c in summary["conditions"]:
        rows = [("ALL", c["overall"])]
        if len(conditions) <= 3:  # Few conditions: also show each command.
            rows = list(c["commands"].items()) + rows
        for name, s in rows:
            t_fall = s["median_fall_time_s"]
            t_fall = "     -" if t_fall is None else f"{t_fall:6.2f}"
            print(f"{c['payload_mass_g']:6.0f} {c['payload_dx_mm']:6.0f} {name:13s} {s['fall_rate']:6.3f} {t_fall} "
                  f"{s['rmse_xy']:8.3f} {s['rmse_yaw']:8.3f} {s['mean_vx']:6.2f} {s['mean_vy']:6.2f} {s['mean_wz']:6.2f}")
    print(f"NaN episodes: {summary['nan_episodes']}. Wall time {summary['wall_time_s']} s. Wrote {out}")


if __name__ == "__main__":
    main()
