"""Payload: a small rigid box fixed to the trunk of the robot.

The payload is a separate body ("payload") that is welded to the trunk
(no joint). Each environment has its own payload mass and position, so one
simulation can test many payloads in parallel.

Position. The mount point is MOUNT_POS in the trunk frame: on top of the
trunk, above the centre of mass (CoM) of the robot in its standing pose
(the whole-robot CoM is at x = 0.6 mm, y = 0, z = 22 mm in the trunk frame).
A payload at the mount point adds weight and raises the CoM, but it does not
move the CoM forward or backward. The offset moves the payload from the mount
point: +x is forward, +y is left.

With robot mass M = 0.737 kg, a payload of mass m at forward offset dx moves
the whole-robot CoM forward by m * dx / (M + m).

A payload of mass 0 changes nothing: the robot is then the same as upstream.
"""

from dataclasses import replace

import mujoco
import torch
from mjlab.managers import EventTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

ROBOT_MASS_KG = 0.737  # Sum of the body masses of the upstream walk model.
PAYLOAD_BODY = "payload"
MOUNT_POS = (0.0, 0.0, 0.05)  # Trunk frame, metres. z = 0.05 is the top of the trunk.
BOX_HALF_SIZE = (0.015, 0.015, 0.010)  # 30 x 30 x 20 mm box.


def add_payload_body(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add the payload body (mass 0) to the trunk. The box is visual only: no contact."""
    body = spec.body("trunk_base").add_body(name=PAYLOAD_BODY, pos=list(MOUNT_POS))
    body.mass = 0.0
    body.inertia = [0.0, 0.0, 0.0]
    body.explicitinertial = True
    body.add_geom(
        name="payload_visual",  # Not "*_collision": the upstream collision rule ignores it.
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(BOX_HALF_SIZE),
        contype=0,
        conaffinity=0,
        rgba=[0.85, 0.2, 0.2, 1.0],
    )
    return spec


def box_inertia(mass: torch.Tensor) -> torch.Tensor:
    """Principal inertia (N, 3) of a uniform box with BOX_HALF_SIZE and mass (N,)."""
    a, b, c = (2.0 * h for h in BOX_HALF_SIZE)  # Full side lengths.
    shape = torch.tensor([b * b + c * c, a * a + c * c, a * a + b * b], dtype=mass.dtype, device=mass.device)
    return mass.unsqueeze(-1) * shape / 12.0


def com_shift_m(mass_kg: float, offset_m: float) -> float:
    """Shift of the whole-robot CoM (m) caused by a payload at a horizontal offset."""
    return mass_kg * offset_m / (ROBOT_MASS_KG + mass_kg)


@requires_model_fields("body_mass", "body_inertia", "body_pos", recompute=RecomputeLevel.set_const)
def set_payload(env, env_ids, mass_kg, offset_m) -> None:
    """Event: write the payload mass, inertia and position of each env into the model.

    mass_kg: one value per env (length num_envs).
    offset_m: one (x, y, z) offset from MOUNT_POS per env (num_envs x 3).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    mass = torch.as_tensor(mass_kg, dtype=torch.float32, device=env.device)
    offset = torch.as_tensor(offset_m, dtype=torch.float32, device=env.device)
    assert mass.shape == (env.num_envs,) and offset.shape == (env.num_envs, 3)

    mj_model = env.sim.mj_model
    ids = [i for i in range(mj_model.nbody) if mj_model.body(i).name.split("/")[-1] == PAYLOAD_BODY]
    assert len(ids) == 1, f"expected one payload body, found {len(ids)}"
    body = ids[0]

    mount = torch.tensor(MOUNT_POS, dtype=torch.float32, device=env.device)
    model = env.sim.model
    model.body_mass[env_ids, body] = mass[env_ids]
    model.body_inertia[env_ids, body] = box_inertia(mass[env_ids])
    model.body_pos[env_ids, body] = mount + offset[env_ids]


def add_payload_to_cfg(cfg, mass_kg: list[float], offset_m: list[tuple[float, float, float]]) -> None:
    """Change an env cfg in place: robot with payload body, and the event that sets it."""
    robot = cfg.scene.entities["robot"]
    base_spec_fn = robot.spec_fn
    cfg.scene.entities["robot"] = replace(robot, spec_fn=lambda: add_payload_body(base_spec_fn()))
    cfg.events["set_payload"] = EventTermCfg(
        func=set_payload,
        mode="startup",
        params={"mass_kg": list(mass_kg), "offset_m": [list(o) for o in offset_m]},
    )
