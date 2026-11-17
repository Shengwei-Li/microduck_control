"""Training environments with a random payload in the beak.

The training env is the upstream velocity task plus two events:

  reset_payload   (at each episode reset) The payload mass of the env goes to 0.
                  The env draws the mass for this episode (uniform in
                  mass_range_kg) and the time when the payload appears
                  (uniform in onset_range_s).
  pickup_payload  (each step) Envs whose time has come get their payload:
                  "the robot picks up the object".

Methods (see TRAIN_TASKS):

  P1   The policy does not see the mass. It must be robust to it.
  P0c  Control: mass range (0, 0). The task is the same as upstream, with the
       same training procedure as P1.
  P2   P1 plus the true payload mass in the observations of the actor and the
       critic. This is the reference: a policy that knows the answer.
  P3   The actor sees the last HISTORY_STEPS observations (it can estimate the
       payload from them). Only the critic sees the true mass.

P2 and P3 change the size of the network input. checkpoint.py changes the P0
checkpoint to the new size, so that the first action is the same as P0.

Speed. After a mass change, the simulator must compute its mass-dependent
constants again for all envs. To do this less often, pickup_payload only acts
each PICKUP_EVERY_STEPS steps. A payload therefore appears up to 0.2 s late.
"""

from copy import deepcopy
from dataclasses import dataclass, replace

import torch
from mjlab.managers import EventTermCfg, ObservationTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from payload_duck.payload import add_payload_body, payload_body_id, write_payload

BASE_TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"
PICKUP_EVERY_STEPS = 10  # 10 steps = 0.2 s.
MASS_OBS_SCALE_KG = 0.100  # The mass observation is mass / 100 g: 0 to 1 in training.
HISTORY_STEPS = 25  # 25 steps = 0.5 s.
ONSET_RANGE_S = (0.0, 10.0)  # The episode is 20 s.


@dataclass(frozen=True)
class Method:
    mass_range_kg: tuple[float, float]  # Payload mass range in training.
    mass_obs: tuple[str, ...] = ()  # Observation groups that get the true mass.
    actor_history: int = 0  # Number of observations that the actor sees. 0 = only the newest.


TRAIN_TASKS = {
    "PayloadDuck-P1": Method((0.0, 0.100)),
    "PayloadDuck-P0c": Method((0.0, 0.0)),
    "PayloadDuck-P2": Method((0.0, 0.100), mass_obs=("actor", "critic")),
    "PayloadDuck-P3": Method((0.0, 0.100), mass_obs=("critic",), actor_history=HISTORY_STEPS),
}


def _buffers(env):
    """Per-env payload state. It is stored on the env object."""
    if not hasattr(env, "payload_mass_kg"):
        env.payload_mass_kg = torch.zeros(env.num_envs, device=env.device)  # Mass of this episode.
        env.payload_onset_s = torch.zeros(env.num_envs, device=env.device)
        env.payload_loaded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.payload_mass_kg, env.payload_onset_s, env.payload_loaded


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def reset_payload(env, env_ids, mass_range_kg, onset_range_s, mount="mouth") -> None:
    """Reset event: remove the payload and draw the mass and onset time of the new episode."""
    mass, onset, loaded = _buffers(env)
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    lo, hi = mass_range_kg
    mass[env_ids] = lo + (hi - lo) * torch.rand(n, device=env.device)
    lo, hi = onset_range_s
    onset[env_ids] = lo + (hi - lo) * torch.rand(n, device=env.device)
    loaded[env_ids] = False
    write_payload(env, env_ids, torch.zeros(n, device=env.device), mount)


# No automatic recompute here: the event manager would do it after each step,
# and after the env has already computed the body positions for the observations.
@requires_model_fields("body_mass", "body_inertia")
def pickup_payload(env, env_ids, mount="mouth") -> None:
    """Step event: give the payload to the envs whose onset time has passed."""
    if env.common_step_counter % PICKUP_EVERY_STEPS != 0 or not hasattr(env, "payload_mass_kg"):
        return
    mass, onset, loaded = _buffers(env)
    due = (~loaded & (env.episode_length_buf * env.step_dt >= onset)).nonzero().flatten()
    if len(due) == 0:
        return
    write_payload(env, due, mass[due], mount)
    loaded[due] = True
    # Same two steps as payload.load(): constants, then body positions from the state.
    env.sim.recompute_constants(RecomputeLevel.set_const)
    env.sim.forward()


def make_train_env_cfg(base_cfg, mass_range_kg, onset_range_s=ONSET_RANGE_S, mount="mouth"):
    """Return a copy of the upstream env cfg with the payload body and the two payload events."""
    cfg = deepcopy(base_cfg)
    robot = cfg.scene.entities["robot"]
    base_spec_fn = robot.spec_fn
    cfg.scene.entities["robot"] = replace(robot, spec_fn=lambda: add_payload_body(base_spec_fn(), mount))
    cfg.events["reset_payload"] = EventTermCfg(
        func=reset_payload,
        mode="reset",
        params={"mass_range_kg": tuple(mass_range_kg), "onset_range_s": tuple(onset_range_s), "mount": mount},
    )
    cfg.events["pickup_payload"] = EventTermCfg(func=pickup_payload, mode="step", params={"mount": mount})
    return cfg


def payload_mass(env) -> torch.Tensor:
    """Observation: the payload mass that the robot holds now, divided by 100 g. Shape (num_envs, 1).

    It reads the mass from the simulation model. It is 0 before the pickup.
    """
    if not hasattr(env, "payload_body"):
        env.payload_body = payload_body_id(env)
    mass = env.sim.model.body_mass[..., env.payload_body]  # One value for each env, or one for all.
    return mass.reshape(-1, 1).expand(env.num_envs, 1) / MASS_OBS_SCALE_KG


def apply_method_obs(cfg, method: Method) -> None:
    """Change the observations of an env cfg in place for a method (P2, P3).

    The new mass term is the last term of its group. The cfg must have the payload body.
    """
    for group in method.mass_obs:
        cfg.observations[group].terms["payload_mass"] = ObservationTermCfg(func=payload_mass)
    if method.actor_history > 0:
        cfg.observations["actor"].history_length = method.actor_history


def register_train_tasks() -> None:
    """Register the tasks in TRAIN_TASKS with mjlab (same PPO settings and runner as upstream)."""
    import mjlab_microduck.tasks  # noqa: F401  Registers the upstream tasks.
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls, register_mjlab_task

    def env_cfg(method, play):
        cfg = make_train_env_cfg(load_env_cfg(BASE_TASK_ID, play=play), method.mass_range_kg)
        apply_method_obs(cfg, method)
        return cfg

    for task_id, method in TRAIN_TASKS.items():
        if task_id in list_tasks():
            continue
        register_mjlab_task(
            task_id=task_id,
            env_cfg=env_cfg(method, False),
            play_env_cfg=env_cfg(method, True),
            rl_cfg=load_rl_cfg(BASE_TASK_ID),
            runner_cls=load_runner_cls(BASE_TASK_ID),
        )
