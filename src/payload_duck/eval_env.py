"""Evaluation environment: the training environment with fixed conditions.

The training environment changes during training (curriculum) and pushes the
robot at random times. For a fair comparison between policies and payloads,
the evaluation environment removes these sources of variation:

1. No curriculum. The domain randomization ranges are set to the values that
   the curriculum reaches at the end of training.
2. No random pushes. A fall must come from the payload, not from a push.
3. Fixed velocity commands. Each environment gets one command from
   EVAL_COMMANDS and keeps it for the full episode.
4. Head and body pose commands are zero (neutral head).
5. Optional: a payload with a fixed mass and position for each environment
   (see payload.py). It can be there from the start, or appear at a fixed
   time in the episode.

The other domain randomization stays on (mass, joint friction, armature,
IMU mounting error, encoder bias). They are part of the normal operating
conditions of the policy. With the same seed, all policies see the same draws.
"""

from copy import deepcopy

import torch

TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"

# Fixed velocity commands (body frame): name -> (lin_vel_x m/s, lin_vel_y m/s, ang_vel_z rad/s).
# Training range: x in [-0.4, 0.4], y in [-0.3, 0.3], yaw in [-1.0, 1.0].
EVAL_COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward_slow": (0.2, 0.0, 0.0),
    "forward_fast": (0.4, 0.0, 0.0),
    "backward": (-0.3, 0.0, 0.0),
    "turn": (0.0, 0.0, 0.8),
    "forward_turn": (0.3, 0.0, 0.5),
}
# No pure sideways command: the baseline policy stands still for vy <= 0.2 m/s
# (D1 result), so that command cannot show an effect of the payload.

# Final values of the upstream curriculum (microduck_velocity_env_cfg.py).
FINAL_COM_RANGE_M = 0.015
FINAL_HEAD_COM_RANGE_M = 0.01

NEVER_S = 1.0e9  # A resampling time longer than any episode.


def payload_per_env(payloads, envs_per_command: int):
    """Payload of each env: (mass_kg list, offset_m list) for a list of conditions (mass_kg, dx_m)."""
    block = envs_per_command * len(EVAL_COMMANDS)
    mass = [m for m, _ in payloads for _ in range(block)]
    offset = [(dx, 0.0, 0.0) for _, dx in payloads for _ in range(block)]
    return mass, offset


def make_eval_env_cfg(base_cfg, envs_per_command: int, seed: int, payloads=None,
                      mount: str = "mouth", loaded_at_start: bool = True):
    """Return a copy of the training env cfg with the fixed evaluation conditions.

    payloads: None (upstream robot, no payload body), or a list of conditions
    (mass_kg, dx_m). Each condition gets a full block of all commands.
    Env layout: env i has condition i // (envs_per_command * n_commands) and
    command (i // envs_per_command) % n_commands.
    loaded_at_start: False starts all envs with mass 0. The caller then writes
    the payload during the episode with payload.load().
    """
    cfg = deepcopy(base_cfg)
    n_conditions = 1 if payloads is None else len(payloads)
    block = envs_per_command * len(EVAL_COMMANDS)
    cfg.scene.num_envs = block * n_conditions
    cfg.seed = seed

    if payloads is not None:
        from payload_duck.payload import add_payload_to_cfg

        mass, offset = payload_per_env(payloads, envs_per_command)
        if not loaded_at_start:
            mass = [0.0] * len(mass)
        add_payload_to_cfg(cfg, mass, offset, mount)

    # 1. No curriculum; domain randomization at its final range.
    cfg.curriculum = {}
    if "randomize_com" in cfg.events:
        cfg.events["randomize_com"].params["ranges"] = (-FINAL_COM_RANGE_M, FINAL_COM_RANGE_M)
    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["ranges"] = (
            -FINAL_HEAD_COM_RANGE_M,
            FINAL_HEAD_COM_RANGE_M,
        )

    # 2. No random pushes.
    cfg.events.pop("push_robot", None)

    # 3. Velocity command: never resampled, no standing or turn-in-place envs.
    #    The values are written by apply_commands() after each reset.
    twist = cfg.commands["twist"]
    twist.resampling_time_range = (NEVER_S, NEVER_S)
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0

    # 4. Neutral head and body pose commands.
    for name in ("head_pose", "body_pose"):
        pose = cfg.commands[name]
        pose.resampling_time_range = (NEVER_S, NEVER_S)
        pose.ranges = tuple((0.0, 0.0) for _ in pose.ranges)

    return cfg


def command_table(envs_per_command: int, device, n_conditions: int = 1) -> tuple[torch.Tensor, list[str]]:
    """Commands for all envs, in blocks of envs_per_command, repeated for each condition."""
    names = list(EVAL_COMMANDS)
    values = torch.tensor([EVAL_COMMANDS[n] for n in names], dtype=torch.float32, device=device)
    table = values.repeat_interleave(envs_per_command, dim=0).repeat(n_conditions, 1)
    labels = [n for n in names for _ in range(envs_per_command)] * n_conditions
    return table, labels


def apply_commands(env, table: torch.Tensor) -> None:
    """Write the fixed velocity commands into the command term of the env."""
    term = env.command_manager.get_term("twist")
    term.vel_command_b[:] = table
    term.vel_command_w[:] = table
    # Clear the flags that make the command term change the command each step.
    term.is_standing_env[:] = False
    term.is_heading_env[:] = False
    term.is_world_env[:] = False
    term.is_forward_env[:] = False
