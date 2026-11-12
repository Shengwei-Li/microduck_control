"""Training environments with a random payload in the beak.

The training env is the upstream velocity task plus two events:

  reset_payload   (at each episode reset) The payload mass of the env goes to 0.
                  The env draws the mass for this episode (uniform in
                  mass_range_kg) and the time when the payload appears
                  (uniform in onset_range_s).
  pickup_payload  (each step) Envs whose time has come get their payload:
                  "the robot picks up the object".

The policy does not see the mass. It must be robust to it (method P1).
With mass_range_kg = (0, 0) the task is the same as upstream; we use it as a
control with the same training procedure (P0c).

Speed. After a mass change, the simulator must compute its mass-dependent
constants again for all envs. To do this less often, pickup_payload only acts
each PICKUP_EVERY_STEPS steps. A payload therefore appears up to 0.2 s late.
"""

from copy import deepcopy
from dataclasses import replace

import torch
from mjlab.managers import EventTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from payload_duck.payload import add_payload_body, write_payload

BASE_TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"
PICKUP_EVERY_STEPS = 10  # 10 steps = 0.2 s.

# Task name -> payload mass range in training (kg).
TRAIN_TASKS = {
    "PayloadDuck-P1": (0.0, 0.100),  # Domain randomization of the payload mass.
    "PayloadDuck-P0c": (0.0, 0.0),  # Control: same procedure, no payload.
}
ONSET_RANGE_S = (0.0, 10.0)  # The episode is 20 s.


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


def register_train_tasks() -> None:
    """Register the tasks in TRAIN_TASKS with mjlab (same PPO settings and runner as upstream)."""
    import mjlab_microduck.tasks  # noqa: F401  Registers the upstream tasks.
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls, register_mjlab_task

    for task_id, mass_range in TRAIN_TASKS.items():
        if task_id in list_tasks():
            continue
        register_mjlab_task(
            task_id=task_id,
            env_cfg=make_train_env_cfg(load_env_cfg(BASE_TASK_ID), mass_range),
            play_env_cfg=make_train_env_cfg(load_env_cfg(BASE_TASK_ID, play=True), mass_range),
            rl_cfg=load_rl_cfg(BASE_TASK_ID),
            runner_cls=load_runner_cls(BASE_TASK_ID),
        )
