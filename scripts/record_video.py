"""Record a short video of a policy under the evaluation conditions.

One robot per command in EVAL_COMMANDS. The video shows each robot in turn for
a few seconds, with a caption: the command and the mean measured velocity in
that clip. Rendering is offscreen (EGL), so no display is necessary.

Run it from the upstream folder (same as evaluate.py):
    cd ~/projectD/third_party/microduck_rl
    uv run --locked python /mnt/d/EE_study/mphil_ready/projectD_rl_payload_biped/scripts/record_video.py \
        --checkpoint logs/rsl_rl/velocity/<run>/model_2999.pt --out <file>.mp4
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # Must be set before mujoco is imported.

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402
from payload_duck.eval_env import EVAL_COMMANDS, apply_commands, command_table  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="output .mp4 file")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--start-s", type=float, default=3.0, help="clip start (skip the first steps)")
    p.add_argument("--clip-s", type=float, default=6.0, help="clip length per command")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--payload-mass-g", type=float, default=None, help="payload mass (g); default: no payload")
    p.add_argument("--payload-dx-mm", type=float, default=0.0, help="payload forward offset (mm)")
    p.add_argument("--payload-mount", choices=["mouth", "trunk"], default="mouth")
    p.add_argument("--payload-onset-s", type=float, default=5.0, help="time when the payload appears; 0 = from the start")
    return p.parse_args()


def set_payload_visible(env, visible: bool) -> None:
    """Show or hide the payload shape in the rendered frames (all envs)."""
    import mujoco

    model = env.sim.mj_model
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""  # Some geoms have no name.
        if name.endswith("payload_visual"):
            model.geom_rgba[i, 3] = 1.0 if visible else 0.0


def caption(frame, lines):
    """Draw text lines in the top left corner of an RGB frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    y = 8
    for line in lines:
        draw.text((10, y), line, fill=(255, 255, 255), font=font)
        y += 24
    return np.asarray(img)


def main():
    args = parse_args()
    args.envs_per_command = 1  # One robot per command; build() reads this.
    args.payloads = evaluate.payload_conditions(
        None if args.payload_mass_g is None else [args.payload_mass_g], [args.payload_dx_mm])
    where = "in the beak" if args.payload_mount == "mouth" else f"on the trunk, {args.payload_dx_mm:+.0f} mm forward"
    payload_text = None if args.payloads is None else f"payload {args.payload_mass_g:.0f} g {where}"
    torch.manual_seed(args.seed)
    viewer = {
        "width": args.width,
        "height": args.height,
        "distance": 0.9,  # The robot is about 0.25 m tall.
        "elevation": -15.0,
        "azimuth": 135.0,
        "max_extra_envs": 0,  # Show only the tracked robot.
        "enable_shadows": False,  # Shadows give artifacts on the large ground plane.
    }
    env, venv, policy = evaluate.build(args, viewer=viewer)
    renderer = env._offline_renderer  # The camera follows renderer._cfg.env_idx.
    table, labels = command_table(1, args.device)
    robot = env.command_manager.get_term("twist").robot

    dt = env.step_dt
    t0 = round(args.start_s / dt)
    t1 = t0 + round(args.clip_s / dt)
    frames = {name: [] for name in labels}
    vel = {name: [] for name in labels}
    load_step, load_fn = evaluate.payload_loader(env, args)
    if load_step is not None:
        set_payload_visible(env, False)
    loaded_from = t0 if load_step is None else max(load_step, t0)  # First step with the final payload.

    venv.reset()
    apply_commands(env, table)
    obs = venv.get_observations()
    with torch.inference_mode():
        for t in range(t1):
            if t == load_step:
                load_fn()
                set_payload_visible(env, True)
            obs, _, dones, _ = venv.step(policy(obs))
            apply_commands(env, table)
            if dones.any():
                print(f"[WARN] env(s) {dones.nonzero().flatten().tolist()} ended at t={t * dt:.2f} s")
            if t < t0:
                continue
            v = torch.cat([robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]], dim=1)
            for i, name in enumerate(labels):
                renderer._cfg.env_idx = i
                frames[name].append(env.render())
                vel[name].append(v[i].cpu().numpy())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = round(1.0 / dt)
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=16) as w:
        for name in labels:
            c = EVAL_COMMANDS[name]
            m = np.mean(vel[name][loaded_from - t0:], axis=0)  # Mean with the final payload.
            lines = [
                f"{name}",
                f"command   vx {c[0]:+.2f}  vy {c[1]:+.2f} m/s  wz {c[2]:+.2f} rad/s",
                f"measured  vx {m[0]:+.2f}  vy {m[1]:+.2f} m/s  wz {m[2]:+.2f} rad/s",
            ]
            for k, frame in enumerate(frames[name]):
                first = []
                if payload_text is not None:
                    first = [payload_text if t0 + k >= loaded_from else "no payload"]
                w.append_data(caption(frame, first + lines))
    print(f"Wrote {out} ({len(labels)} clips x {args.clip_s} s at {fps} fps)")


if __name__ == "__main__":
    main()
