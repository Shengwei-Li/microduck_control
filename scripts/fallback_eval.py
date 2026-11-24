"""Closed-loop test of a monitor with a fallback: after the alarm, the robot stands (D4).

Run it from the upstream folder:
    cd ~/projectD/third_party/microduck_rl
    uv run --locked python <this repo>/scripts/fallback_eval.py [--task PayloadDuck-P3] \\
        --checkpoint logs/rsl_rl/velocity/<run>/model_3998.pt --seed 1002 \\
        --payload-mass-g 0 20 40 60 80 100 120 150 200 \\
        --monitors <monitor_eval file>.json --dynamics <model>.pt --mode residual --out <file>.json

The episodes are the same as in evaluate.py. Modes:
  none      no fallback. The monitors run and their alarms are stored.
  neck      the neck monitor acts: at its alarm, this robot does the fallback.
  residual  the residual monitor acts in the same way.
  onset     reference: all robots get the zero command when the payload appears.
Fallbacks (--fallback): stand (zero velocity command to the end of the episode)
or drop (the robot lets go of the payload: its mass goes to zero, with no delay).
The monitors use a plain threshold. Scales and thresholds come from --monitors
(monitor_eval.py), so they are fixed before this run. The monitors are active
from --from-s to --to-s, the same window as in monitor_eval.py.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate  # noqa: E402
from monitor_collect import newest_frame_columns  # noqa: E402
from payload_duck import monitor, payload, train_env  # noqa: E402
from payload_duck.dynamics import Dynamics  # noqa: E402
from payload_duck.eval_env import EVAL_COMMANDS, apply_commands, command_table, payload_per_env  # noqa: E402

SIGNALS = ("neck", "residual")


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--mode", choices=["none", "neck", "residual", "onset"], required=True)
    extra.add_argument("--fallback", choices=["stand", "drop"], default="stand",
                       help="what the robot does at the alarm: zero velocity command, or let go of the payload")
    extra.add_argument("--monitors", required=True)
    extra.add_argument("--dynamics", required=True)
    extra.add_argument("--from-s", type=float, default=3.0)
    extra.add_argument("--to-s", type=float, default=13.0)
    own, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = evaluate.parse_args()
    assert args.payloads is not None and args.payload_onset_s > 0.0

    torch.manual_seed(args.seed)
    env, venv, policy = evaluate.build(args)
    dev = args.device
    settings = json.loads(Path(own.monitors).read_text())["monitors"]
    settings = {s: settings[s + "_plain"] for s in SIGNALS}
    model = Dynamics()
    model.load_state_dict(torch.load(own.dynamics, map_location="cpu"))
    model = model.to(dev).eval()

    manager = env.observation_manager
    steps = train_env.TRAIN_TASKS[args.task].actor_history if args.task else 0
    names = list(manager.active_terms["actor"])
    dims = [d[0] // max(steps, 1) for d in manager.group_obs_term_dim["actor"]]
    if names[-1] == "payload_mass":
        names, dims = names[:-1], dims[:-1]
    cols = torch.tensor(newest_frame_columns(dims, steps), device=dev)
    terms_of = monitor.term_slices(names, dims)
    robot = env.command_manager.get_term("twist").robot

    table, labels = command_table(args.envs_per_command, dev, len(args.payloads))
    table = table.clone()
    load_step, load_fn = evaluate.payload_loader(env, args)
    terms = env.termination_manager
    first, last = round(own.from_s / env.step_dt), round(own.to_s / env.step_dt)

    n = env.num_envs
    alarm = {s: torch.full((n,), -1, dtype=torch.long, device=dev) for s in SIGNALS}
    fall_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    done_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    prev = None
    venv.reset()
    apply_commands(env, table)
    obs = venv.get_observations()
    with torch.inference_mode():
        for t in range(env.max_episode_length):
            if t == load_step:
                load_fn()
                if own.mode == "onset":
                    table[:] = 0.0
                    apply_commands(env, table)
            action = policy(obs)
            if first <= t < last:
                frame = obs["actor"][:, cols]
                signal = {"neck": monitor.neck_error(frame, terms_of, robot.joint_names)}
                if prev is not None:
                    signal["residual"] = model.residual(prev[0], prev[1], frame)
                prev = (frame, action)
                for s, x in signal.items():
                    z = (x - settings[s]["mu"]) / settings[s]["sigma"]
                    hit = (z > settings[s]["threshold"]) & (done_step < 0) & (alarm[s] < 0)
                    alarm[s][hit] = t
                    if own.mode == s and own.fallback == "stand":
                        table[hit] = 0.0  # The command stays zero to the end.
                    elif own.mode == s and hit.any():
                        payload.drop(env, torch.nonzero(hit)[:, 0], args.payload_mount)
            obs, _, dones, _ = venv.step(action)
            apply_commands(env, table)
            running = done_step < 0
            fall_step[running & terms.get_term("fell_over")] = t
            done_step[running & dones.bool()] = t

    mass_kg, _ = payload_per_env(args.payloads, args.envs_per_command)
    mass = np.array(mass_kg) * 1000.0
    fell = fall_step.cpu().numpy() >= 0
    alarms = {s: a.cpu().numpy() for s, a in alarm.items()}
    moving = np.array(labels) != "stand"  # For robots with the stand command, the fallback changes nothing.
    result = {"mode": own.mode, "fallback": own.fallback, "seed": args.seed, "task": args.task, "checkpoint": str(Path(args.checkpoint).resolve()),
              "window_s": [own.from_s, own.to_s], "commands": list(EVAL_COMMANDS), "by_mass": {}}
    print(f"mode {own.mode}: mass | fall rate | alarm rate neck, residual")
    for m in sorted(set(mass.tolist())):
        k = mass == m
        row = {"episodes": int(k.sum()), "fall_rate": float(fell[k].mean()), "fall_rate_moving": float(fell[k & moving].mean())}
        for s in SIGNALS:
            row[f"alarm_rate_{s}"] = float((alarms[s][k] >= 0).mean())
        result["by_mass"][f"{m:.0f}"] = row
        print(f"  {m:4.0f} | {row['fall_rate']:.3f} | {row['alarm_rate_neck']:.3f} {row['alarm_rate_residual']:.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
