"""Train the one-step model for the residual monitor (D4).

    python dynamics_train.py --data <monitor_collect file>.npz --out <model>.pt

Uses the episodes with a payload inside the training range (0 to 100 g).
One robot in ten is kept apart to check the model (validation).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from payload_duck.dynamics import MEASURED, Dynamics, valid_frames  # noqa: E402

MAX_MASS_G = 100.0


def transitions(d, robots):
    """Inputs x (n, 75) and targets y (n, 34) of all valid steps of the chosen robots."""
    frame = d["frame"][:, robots].astype(np.float32)
    action = d["action"][:, robots].astype(np.float32)
    valid = valid_frames(len(frame), int(d["first_step"]), d["done_step"][robots])
    ok = valid[1:] & valid[:-1]
    x = np.concatenate([frame[:-1], action[:-1]], axis=-1)[ok]
    y = (frame[1:, :, :MEASURED] - frame[:-1, :, :MEASURED])[ok]
    return torch.from_numpy(x), torch.from_numpy(y)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    d = np.load(args.data)
    robots = np.flatnonzero(d["mass_g"] <= MAX_MASS_G)
    held = robots[::10]
    train = np.setdiff1d(robots, held)
    x, y = (t.to(args.device) for t in transitions(d, train))
    xv, yv = (t.to(args.device) for t in transitions(d, held))

    model = Dynamics(n_in=x.shape[1]).to(args.device)
    model.x_mean, model.x_std = x.mean(0), x.std(0) + 1e-6
    model.y_mean, model.y_std = y.mean(0), y.std(0) + 1e-6
    target, target_v = (y - model.y_mean) / model.y_std, (yv - model.y_mean) / model.y_std

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for epoch in range(args.epochs):
        order = torch.randperm(len(x), device=args.device)
        for i in range(0, len(x), 4096):
            idx = order[i:i + 4096]
            loss = (model(x[idx]) - target[idx]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        with torch.inference_mode():
            val = (model(xv) - target_v).pow(2).mean().item()
        print(f"epoch {epoch}: validation error {val:.4f} (1.0 = always predict the mean change)")

    with torch.inference_mode():
        model.err_std = (model(x) - target).std(0) + 1e-6
        res = ((model(xv) - target_v) / model.err_std).pow(2).mean(-1).sqrt()
    print(f"train steps {len(x)}, validation steps {len(xv)}; validation residual mean {res.mean():.3f}, 99 % {res.quantile(0.99):.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
