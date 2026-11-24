"""Still images for the report: one robot before the payload and with the payload.

Run it from the upstream folder:
    cd ~/projectD/third_party/microduck_rl
    uv run --locked python <this repo>/report/setup_figure.py --checkpoint <P1 run>/model_3998.pt

Writes report/figures/setup.png: the same robot (command: stand) at 4.5 s
without payload and at 9 s with the payload.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate  # noqa: E402
from payload_duck.eval_env import apply_commands, command_table  # noqa: E402
from record_video import set_payload_visible  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mass-g", type=float, default=100.0)
    p.add_argument("--azimuth", type=float, default=90.0)
    p.add_argument("--elevation", type=float, default=-24.0)
    p.add_argument("--distance", type=float, default=0.7)
    p.add_argument("--crop", type=int, nargs=4, default=[160, 760, 130, 730], metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"))
    p.add_argument("--out", default=str(ROOT / "report" / "figures" / "setup.png"))
    a = p.parse_args()

    args = argparse.Namespace(checkpoint=a.checkpoint, seed=1000, envs_per_command=1, device="cuda:0", task=None,
                              payloads=[(a.mass_g / 1000.0, 0.0)], payload_mount="mouth", payload_onset_s=5.0)
    torch.manual_seed(args.seed)
    viewer = {"width": 900, "height": 900, "distance": a.distance, "elevation": a.elevation, "azimuth": a.azimuth,
              "max_extra_envs": 0, "enable_shadows": False}
    env, venv, policy = evaluate.build(args, viewer=viewer)
    env._offline_renderer._cfg.env_idx = 0  # The robot with the command "stand".
    table, _ = command_table(1, args.device)
    load_step, load_fn = evaluate.payload_loader(env, args)
    set_payload_visible(env, False)
    shots = {round(4.5 / env.step_dt): None, round(9.0 / env.step_dt): None}

    venv.reset()
    apply_commands(env, table)
    obs = venv.get_observations()
    with torch.inference_mode():
        for t in range(max(shots) + 1):
            if t == load_step:
                load_fn()
                set_payload_visible(env, True)
            if t in shots:
                shots[t] = env.render()
            obs, _, _, _ = venv.step(policy(obs))
            apply_commands(env, table)

    top, bottom, x0, x1 = a.crop  # Crop the margins of the 900 x 900 render.
    left, right = (shots[k][top:bottom, x0:x1] for k in sorted(shots))
    gap = np.full((left.shape[0], 12, 3), 255, dtype=np.uint8)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([left, gap, right], axis=1)).save(a.out)
    print("wrote", a.out)
    env.close()


if __name__ == "__main__":
    main()
