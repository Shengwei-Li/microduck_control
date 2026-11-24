"""Payload: a small rigid object that the robot carries.

The payload is a separate body ("payload") that is welded to the robot
(no joint). Each environment has its own payload mass and position, so one
simulation can test many payloads in parallel.

Mounts (where the payload is attached):

  "mouth"  Main setting. The robot holds a thin rod in its beak, and the
           payload is a ball at the end of the rod. The rod starts at the tip
           of the beak (the upstream site "mouth_tip" on the head). The ball
           is 78 mm in front of the tip. The beak is far in front of the feet,
           so a small mass moves the centre of mass (CoM) of the robot forward.
  "trunk"  Control setting. The payload is a box on a post, 50 mm above the top
           of the trunk. It is above the whole-robot CoM (x = 0.6 mm, y = 0 in
           the trunk frame). It adds weight but does not move the CoM forward.

The rod and the post have no mass and no contact. They only show how the
robot holds the payload.

The offset moves the payload centre (parent body frame).
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
    anchor: tuple[float, float, float]  # Where the robot holds the rod, in the parent frame (m).
    pos: tuple[float, float, float]  # Centre of the payload (the mass is here), in the parent frame (m).
    geom: int  # mujoco geom type of the payload shape.
    size: tuple[float, ...]  # Sphere: (radius,). Box: half sizes (x, y, z).
    rgba: tuple[float, float, float, float]

    def unit_inertia(self) -> tuple[float, float, float]:
        """Principal inertia of the uniform shape for a mass of 1 kg (kg m^2 per kg)."""
        if self.geom == mujoco.mjtGeom.mjGEOM_SPHERE:
            i = 0.4 * self.size[0] ** 2  # Solid sphere: 2/5 m r^2.
            return (i, i, i)
        a, b, c = (2.0 * h for h in self.size)  # Box: full side lengths.
        return ((b * b + c * c) / 12.0, (a * a + c * c) / 12.0, (a * a + b * b) / 12.0)


ROD_RADIUS_M = 0.002
ROD_RGBA = (0.15, 0.15, 0.15, 1.0)

MOUNTS = {
    # anchor = the upstream site "mouth_tip" in the frame of the head body "jaw_soft".
    # In this frame, -z is forward. The payload centre is at 2 x anchor.
    "mouth": Mount("jaw_soft", (-0.00809334, 0.0, -0.0777383), (-0.01618668, 0.0, -0.1554766),
                   mujoco.mjtGeom.mjGEOM_SPHERE, (0.015,), (0.1, 0.35, 0.9, 1.0)),  # Blue: the robot is white and orange.
    # z = 0.05 is the top of the trunk. 30 x 30 x 20 mm box.
    "trunk": Mount("trunk_base", (0.0, 0.0, 0.05), (0.0, 0.0, 0.10),
                   mujoco.mjtGeom.mjGEOM_BOX, (0.015, 0.015, 0.010), (0.85, 0.2, 0.2, 1.0)),
}


def add_payload_body(spec: mujoco.MjSpec, mount: str = "mouth") -> mujoco.MjSpec:
    """Add the payload body (mass 0) to the robot. The shapes are visual only: no contact."""
    mt = MOUNTS[mount]
    body = spec.body(mt.parent).add_body(name=PAYLOAD_BODY, pos=list(mt.pos))
    body.mass = 0.0
    body.inertia = [0.0, 0.0, 0.0]
    # The mass is at the body origin. Without this line, MuJoCo sets ipos = pos.
    body.ipos = [0.0, 0.0, 0.0]
    body.explicitinertial = True
    # The names end with "payload_visual" (the video script hides them by this
    # name) and not with "_collision" (the upstream collision rule ignores them).
    body.add_geom(name="payload_visual", type=mt.geom, size=list(mt.size),
                  contype=0, conaffinity=0, rgba=list(mt.rgba))
    # Rod from the anchor to the payload centre, in the payload body frame.
    to_anchor = [a - p for a, p in zip(mt.anchor, mt.pos)]
    body.add_geom(name="rod_payload_visual", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  fromto=to_anchor + [0.0, 0.0, 0.0], size=[ROD_RADIUS_M, 0.0, 0.0],
                  contype=0, conaffinity=0, rgba=list(ROD_RGBA))
    return spec


def com_shift_m(mass_kg: float, offset_m: float) -> float:
    """Shift of the whole-robot CoM (m) caused by a payload at a horizontal offset."""
    return mass_kg * offset_m / (ROBOT_MASS_KG + mass_kg)


@requires_model_fields("body_mass", "body_inertia", "body_pos", recompute=RecomputeLevel.set_const)
def set_payload(env, env_ids, mass_kg, offset_m, mount="mouth") -> None:
    """Event: write the payload mass, inertia and position of each env into the model.

    mass_kg: one value per env (length num_envs).
    offset_m: one (x, y, z) offset of the payload centre per env (num_envs x 3).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    mass = torch.as_tensor(mass_kg, dtype=torch.float32, device=env.device)
    offset = torch.as_tensor(offset_m, dtype=torch.float32, device=env.device)
    assert mass.shape == (env.num_envs,) and offset.shape == (env.num_envs, 3)
    write_payload(env, env_ids, mass[env_ids], mount, offset[env_ids])


def payload_body_id(env) -> int:
    """Index of the payload body in the model (body names have an entity prefix)."""
    mj_model = env.sim.mj_model
    ids = [i for i in range(mj_model.nbody) if mj_model.body(i).name.split("/")[-1] == PAYLOAD_BODY]
    assert len(ids) == 1, f"expected one payload body, found {len(ids)}"
    return ids[0]


def write_payload(env, env_ids, mass, mount="mouth", offset=None) -> None:
    """Write mass and inertia (and the position, if offset is given) for the envs env_ids.

    mass: (len(env_ids),) in kg. offset: (len(env_ids), 3) in m, or None to keep the position.
    This only writes the model. The caller must make sure that the constants
    that depend on the mass are computed again (see load()).
    """
    body = payload_body_id(env)
    mt = MOUNTS[mount]
    unit_inertia = torch.tensor(mt.unit_inertia(), dtype=torch.float32, device=env.device)
    model = env.sim.model
    model.body_mass[env_ids, body] = mass
    model.body_inertia[env_ids, body] = mass.unsqueeze(-1) * unit_inertia
    if offset is not None:
        pos = torch.tensor(mt.pos, dtype=torch.float32, device=env.device)
        model.body_pos[env_ids, body] = pos + offset


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


def drop(env, env_ids, mount="mouth") -> None:
    """The robots env_ids let go of the payload now: its mass goes to zero.

    Same steps as load(). The release has no delay: this is the best case.
    """
    zero = torch.zeros(len(env_ids), dtype=torch.float32, device=env.device)
    write_payload(env, env_ids, zero, mount)
    env.sim.recompute_constants(RecomputeLevel.set_const)
    env.sim.forward()
