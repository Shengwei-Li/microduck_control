"""Learned one-step model of the robot under one policy (D4, residual monitor).

Input: the frame of this step (61 values: what the robot measures, the last
action, the commands) and the action of this step (14 values).
Output: the change of the measured part of the frame (the first 34 values:
angular velocity, gravity direction, joint positions, joint velocities) to
the next step.

The model is trained on normal data (payload inside the training range).
The residual of one step is the size of the prediction error, in units of the
error on the training data. It is high when the robot does not move as the
model expects.
"""

import numpy as np
import torch
from torch import nn

MEASURED = 34  # base_ang_vel 3, projected_gravity 3, joint_pos 14, joint_vel 14


class Dynamics(nn.Module):
    def __init__(self, n_in=75, n_out=MEASURED, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, n_out))
        # Scales of the inputs and of the targets, and the error scale on the training data.
        self.register_buffer("x_mean", torch.zeros(n_in))
        self.register_buffer("x_std", torch.ones(n_in))
        self.register_buffer("y_mean", torch.zeros(n_out))
        self.register_buffer("y_std", torch.ones(n_out))
        self.register_buffer("err_std", torch.ones(n_out))

    def forward(self, x):
        """Standardized change of the measured values, from the raw input."""
        return self.net((x - self.x_mean) / self.x_std)

    def residual(self, frame, action, next_frame):
        """One number per robot: RMS over the 34 measured values of error / err_std."""
        x = torch.cat([frame, action], dim=-1)
        target = (next_frame[..., :MEASURED] - frame[..., :MEASURED] - self.y_mean) / self.y_std
        err = (self(x) - target) / self.err_std
        return err.pow(2).mean(dim=-1).sqrt()


def valid_frames(n_steps, first_step, done_step):
    """(steps, robots) True where the stored frame is from the first episode of the robot."""
    t = first_step + np.arange(n_steps)[:, None]
    return (done_step[None, :] < 0) | (t <= done_step[None, :])


def residual_signal(model, frame, action, valid, device="cpu", block=50):
    """Residual for stored data: (steps, robots), NaN where a step or the step before it is not valid.

    The residual at step t uses the frames t-1 and t: it is known when frame t arrives.
    """
    T, N, _ = frame.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    model = model.to(device).eval()
    with torch.inference_mode():
        for s in range(0, T - 1, block):
            e = min(s + block, T - 1)
            f = torch.as_tensor(frame[s:e + 1].astype(np.float32), device=device)
            a = torch.as_tensor(action[s:e].astype(np.float32), device=device)
            out[s + 1:e + 1] = model.residual(f[:-1], a, f[1:]).cpu().numpy()
    out[1:][~(valid[1:] & valid[:-1])] = np.nan
    return out
