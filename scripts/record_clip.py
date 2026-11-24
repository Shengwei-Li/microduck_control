"""Record one clip for the demonstration video: one robot, one policy, one condition.

Run it from the upstream folder:
    cd ~/projectD/third_party/microduck_rl
    uv run --locked python <this repo>/scripts/record_clip.py [--task PayloadDuck-P3] \\
        --checkpoint <run>/model_3998.pt --command forward_fast --payload-mass-g 200 \\
        --mode residual --fallback drop --monitors <file>.json --dynamics <model>.pt \\
        --expect walk --out <folder>/<name>

Writes <name>.mp4 (no text) and <name>.json (what happened: fall, alarm, mean
velocity). make_demo.py puts the clips side by side and adds the text.

The script records --candidates robots with the same command. --expect names
the more frequent outcome of this condition in the tests (fall or walk). The
clip is the first candidate with that outcome, so that it shows a typical
episode. If no candidate has it, the script stops with an error.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # Must be set before mujoco is imported.

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402
from fallback_eval import OnlineMonitors  # noqa: E402
from payload_duck import payload  # noqa: E402
from payload_duck.eval_env import EVAL_COMMANDS, apply_commands, command_table  # noqa: E402
from record_video import set_payload_visible  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--command", required=True, choices=list(EVAL_COMMANDS))
    p.add_argument("--payload-mass-g", type=float, required=True, help="0 = no payload")
    p.add_argument("--mode", choices=["none", "neck", "residual"], default="none", help="monitor that acts")
    p.add_argument("--fallback", choices=["stand", "drop"], default="stand")
    p.add_argument("--monitors", default=None)
    p.add_argument("--dynamics", default=None)
    p.add_argument("--expect", choices=["fall", "walk"], required=True)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--from-s", type=float, default=3.5)
    p.add_argument("--to-s", type=float, default=10.0)
    p.add_argument("--size", type=int, default=480)
    p.add_argument("--distance", type=float, default=0.8)
    p.add_argument("--elevation", type=float, default=-27.0)
    p.add_argument("--side-deg", type=float, default=110.0, help="camera azimuth relative to the heading of the robot")
    p.add_argument("--out", required=True, help="output path without extension")
    a = p.parse_args()

    args = argparse.Namespace(checkpoint=a.checkpoint, seed=a.seed, envs_per_command=a.candidates, device="cuda:0",
                              task=a.task, payloads=[(a.payload_mass_g / 1000.0, 0.0)], payload_mount="mouth",
                              payload_onset_s=5.0)
    torch.manual_seed(a.seed)
    viewer = {"width": a.size, "height": a.size, "distance": a.distance, "elevation": a.elevation, "azimuth": 90.0,
              "max_extra_envs": 0, "enable_shadows": False}
    env, venv, policy = evaluate.build(args, viewer=viewer)
    dev = args.device
    for name in env.command_manager.active_terms:  # No arrows for the commands in the picture.
        env.command_manager.get_term(name).cfg.debug_vis = False
    renderer = env._offline_renderer  # The camera follows renderer._cfg.env_idx.
    robot = env.command_manager.get_term("twist").robot
    terms = env.termination_manager
    table, labels = command_table(a.candidates, dev)
    table = table.clone()
    ids = [i for i, name in enumerate(labels) if name == a.command]  # The candidates.
    monitors = OnlineMonitors(env, a.task, a.monitors, a.dynamics, dev) if a.monitors else None
    load_step, load_fn = evaluate.payload_loader(env, args)
    dt = env.step_dt
    t0, t1 = round(a.from_s / dt), round(a.to_s / dt)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writers = [imageio.get_writer(f"{out}_cand{k}.mp4", fps=round(1 / dt), codec="libx264", quality=8, macro_block_size=16)
               for k in range(len(ids))]
    n = env.num_envs
    fall_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    done_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    act_step = torch.full((n,), -1, dtype=torch.long, device=dev)  # Step of the fallback.
    dropped = torch.zeros(n, dtype=torch.bool, device=dev)
    vx = [[] for _ in ids]
    frames = [0] * len(ids)
    loaded = False

    venv.reset()
    apply_commands(env, table)
    obs = venv.get_observations()
    with torch.inference_mode():
        for t in range(t1):
            if t == load_step:
                load_fn()
                loaded = a.payload_mass_g > 0
            action = policy(obs)
            if monitors is not None:
                for s, hit in monitors.step(obs, action, t, done_step < 0).items():
                    if s != a.mode or not hit.any():
                        continue
                    act_step[hit] = t
                    if a.fallback == "stand":
                        table[hit] = 0.0
                    else:
                        payload.drop(env, torch.nonzero(hit)[:, 0])
                        dropped |= hit
            if t >= t0:  # Camera at the side of each robot. It follows the heading slowly, so that it does not swing.
                heading = torch.rad2deg(robot.data.heading_w).tolist()
                target = [heading[i] + a.side_deg for i in ids]
                if t == t0:
                    side = target
                side = [s + 0.06 * ((g - s + 180.0) % 360.0 - 180.0) for s, g in zip(side, target)]
            if t >= t0:
                for k, i in enumerate(ids):
                    if done_step[i] >= 0:
                        continue  # The episode of this robot ended: its clip stops here.
                    set_payload_visible(env, loaded and not bool(dropped[i]))
                    renderer._cfg.env_idx = i
                    renderer._cam.azimuth = side[k]  # The renderer makes its camera once: set the camera itself.
                    writers[k].append_data(env.render())
                    frames[k] += 1
                    if t * dt >= 7.0:
                        vx[k].append(float(robot.data.root_link_lin_vel_b[i, 0]))
            obs, _, dones, _ = venv.step(action)
            apply_commands(env, table)
            running = done_step < 0
            fall_step[running & terms.get_term("fell_over")] = t
            done_step[running & dones.bool()] = t
    for w in writers:
        w.close()

    fell = [bool(fall_step[i] >= 0) for i in ids]
    print("candidates fell:", fell)
    want = a.expect == "fall"
    chosen = next((k for k, f in enumerate(fell) if f == want), None)
    for k in range(len(ids)):
        path = Path(f"{out}_cand{k}.mp4")
        if k == chosen:
            path.replace(out.with_suffix(".mp4"))
        else:
            path.unlink()
    if chosen is None:
        raise SystemExit(f"no candidate with the outcome '{a.expect}'")
    i = ids[chosen]

    def time_of(step):
        return None if int(step[i]) < 0 else round((int(step[i]) + 1) * dt, 2)

    info = {
        "task": a.task, "checkpoint": str(Path(a.checkpoint).resolve()), "seed": a.seed, "candidate": chosen,
        "candidates_fell": fell, "command": a.command, "command_value": EVAL_COMMANDS[a.command],
        "payload_mass_g": a.payload_mass_g, "payload_onset_s": 5.0, "mode": a.mode, "fallback": a.fallback,
        "from_s": a.from_s, "fps": round(1 / dt), "frames": frames[chosen],
        "fall_s": time_of(fall_step), "fallback_s": time_of(act_step),
        "alarm_s": {s: time_of(v) for s, v in monitors.alarm.items()} if monitors else None,
        "mean_vx": round(float(np.mean(vx[chosen])), 3) if vx[chosen] else None,
    }
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: info[k] for k in ("candidate", "frames", "fall_s", "fallback_s", "alarm_s", "mean_vx")}))
    env.close()


if __name__ == "__main__":
    main()
