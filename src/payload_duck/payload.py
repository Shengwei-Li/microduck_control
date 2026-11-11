"""Payload: a small rigid object that the robot carries.

The payload is a separate body ("payload") that is welded to the robot
(no joint). Each environment has its own payload mass and position, so one
simulation can test many payloads in parallel.

Mounts (where the payload is attached):

  "mouth"  Main setting. The payload is a ball at the tip of the beak (the
           upstream site "mouth_tip" on the head). The robot holds an object
           of unknown mass. The beak is far in front of the feet, so a small
           mass moves the centre of mass (CoM) of the robot forward.
  "trunk"  Control setting. The payload is a box on top of the trunk, above
           the whole-robot CoM (x = 0.6 mm, y = 0, z = 22 mm in the trunk
           frame). It adds weight but does not move the CoM forward.

The offset moves the payload from the mount point (parent body frame).
With robot mass M = 0.737 kg, a payload of mass m at horizontal distance dx
from the CoM moves the whole-robot CoM by m * dx / (M + m).

A payload of mass 0 changes nothing: the robot is then the same as upstream.

Mid-episode loading. The startup event can write mass 0, and load() writes
the real mass later. This is "the robot picks up the object at time t".
"""

from dataclasses import dataclass, replace

import mujoco
import torch
from mjlab.managers import EventTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

ROBOT_MASS_KG = 0.737  # Sum of the body masses of the upstream walk model.
PAYLOAD_BODY = "payload"


@dataclass(frozen=True)
class Mount:
    parent: str  # Name of the body that carries the payload.
    pos: tuple[float, float, float]  # Mount point in the parent frame (m).
    geom: int  # mujoco geom type of the visual shape.
    size: tuple[float, ...]  # Sphere: (radius,). Box: half sizes (x, y, z).
    rgba: tuple[float, float, float, float]

    def unit_inertia(self) -> tuple[float, float, float]:
        """Principal inertia of the uniform shape for a mass of 1 kg (kg m^2 per kg)."""
        if self.geom == mujoco.mjtGeom.mjGEOM_SPHERE:
            i = 0.4 * self.size[0] ** 2  # Solid sphere: 2/5 m r^2.
            return (i, i, i)
        a, b, c = (2.0 * h for h in self.size)  # Box: full side lengths.
        return ((b * b + c * c) / 12.0, (a * a + c * c) / 12.0, (a * a + b * b) / 12.0)


MOUNTS = {
    # pos = the upstream site "mouth_tip" in the frame of the head body "jaw_soft".
    "mouth": Mount("jaw_soft", (-0.00809334, 0.0, -0.0777383), mujoco.mjtGeom.mjGEOM_SPHERE,
                   (0.015,), (0.1, 0.35, 0.9, 1.0)),  # Blue: the robot is white and orange.
    # z = 0.05 is the top of the trunk. 30 x 30 x 20 mm box.
    "trunk": Mount("trunk_base", (0.0, 0.0, 0.05), mujoco.mjtGeom.mjGEOM_BOX,
                   (0.015, 0.015, 0.010), (0.85, 0.2, 0.2, 1.0)),
}


def add_payload_body(spec: mujoco.MjSpec, mount: str = "mouth") -> mujoco.MjSpec:
    """Add the payload body (mass 0) to the robot. The shape is visual only: no contact."""
    mt = MOUNTS[mount]
    body = spec.body(mt.parent).add_body(name=PAYLOAD_BODY, pos=list(mt.pos))
    body.mass = 0.0
    body.inertia = [0.0, 0.0, 0.0]
    body.explicitinertial = True
    body.add_geom(
        name="payload_visual",  # Not "*_collision": the upstream collision rule ignores it.
        type=mt.geom,
        size=list(mt.size),
        contype=0,
        conaffinity=0,
        rgba=list(mt.rgba),
    )
    return spec


def com_shift_m(mass_kg: float, offset_m: float) -> float:
    """Shift of the whole-robot CoM (m) caused by a payload at a horizontal offset."""
    return mass_kg * offset_m / (ROBOT_MASS_KG + mass_kg)


@requires_model_fields("body_mass", "body_inertia", "body_pos", recompute=RecomputeLevel.set_const)
def set_payload(env, env_ids, mass_kg, offset_m, mount="mouth") -> None:
    """Event: write the payload mass, inertia and position of each env into the model.

    mass_kg: one value per env (length num_envs).
    offset_m: one (x, y, z) offset from the mount point per env (num_envs x 3).
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

    mt = MOUNTS[mount]
    pos = torch.tensor(mt.pos, dtype=torch.float32, device=env.device)
    unit_inertia = torch.tensor(mt.unit_inertia(), dtype=torch.float32, device=env.device)
    model = env.sim.model
    model.body_mass[env_ids, body] = mass[env_ids]
    model.body_inertia[env_ids, body] = mass[env_ids].unsqueeze(-1) * unit_inertia
    model.body_pos[env_ids, body] = pos + offset[env_ids]


def load(env, mass_kg, offset_m, mount="mouth") -> None:
    """Write the payload of all envs now (during an episode).

    An event manager calls set_payload and then recomputes the constants that
    depend on the mass. Here we do the same two steps by hand.

    The recompute step does not change the state (joint positions, velocities).
    But it uses the simulation data as work space and leaves the body positions
    of the zero pose there. forward() computes them again from the state.
    """
    set_payload(env, None, mass_kg, offset_m, mount)
    env.sim.recompute_constants(RecomputeLevel.set_const)
    env.sim.forward()


def add_payload_to_cfg(cfg, mass_kg: list[float], offset_m: list[tuple[float, float, float]],
                       mount: str = "mouth") -> None:
    """Change an env cfg in place: robot with payload body, and the startup event that sets it.

    For mid-episode loading, give mass 0 here and call load() later.
    """
    robot = cfg.scene.entities["robot"]
    base_spec_fn = robot.spec_fn
    cfg.scene.entities["robot"] = replace(robot, spec_fn=lambda: add_payload_body(base_spec_fn(), mount))
    cfg.events["set_payload"] = EventTermCfg(
        func=set_payload,
        mode="startup",
        params={"mass_kg": list(mass_kg), "offset_m": [list(o) for o in offset_m], "mount": mount},
    )
