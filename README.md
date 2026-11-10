# Payload on a small biped: randomize, adapt or monitor?

A small biped (Microduck, about 0.25 m tall) walks with a reinforcement-learning policy.
We put an unknown payload on its trunk. This project asks three questions:

1. At which payload mass and offset does a standard walking policy fail, and how?
2. Which method keeps the robot walking best: domain randomization of the payload,
   privileged payload information, or adaptation from observation history?
3. Can a runtime monitor detect the failure early enough to stop the robot safely?

This work is in simulation only. It uses the
[microduck_rl](https://github.com/pollen-robotics/microduck_rl) task from Pollen Robotics
(MuJoCo Warp through mjlab, PPO from rsl_rl). The upstream code is not part of this
repository. We use it at a fixed commit and do not change it.

## Status

| Stage | Content | Status |
|---|---|---|
| D0 | Environment setup and speed test on one laptop GPU (RTX 5060, 8 GB) | Done |
| D1 | Baseline policy P0 (upstream task, no payload) and evaluation protocol | Done |
| D2 | Payload variants and the failure curve of P0 | Next |
| D3 | Payload randomization, privileged information, history adaptation | Planned |
| D4 | Runtime monitors with CUSUM and conformal thresholds | Planned |

## Baseline result (P0, no payload)

P0 trains for 3000 iterations with 4096 parallel environments (2.4 h). The evaluation
removes the curriculum and the random pushes, and gives each robot one fixed velocity
command for a 20 s episode. 128 robots per command, development seed 1000.

| Command | Value | Fall rate | Mean measured velocity |
|---|---|---|---|
| stand | 0 | 0 | 0.00 m/s |
| forward_slow | vx 0.2 m/s | 0 | 0.11 m/s |
| forward_fast | vx 0.4 m/s | 0 | 0.22 m/s |
| backward | vx -0.3 m/s | 0 | -0.18 m/s |
| sideways | vy 0.2 m/s | 0 | 0.01 m/s |
| turn | wz 0.8 rad/s | 0 | 0.65 rad/s |

P0 does not fall in 896 episodes. But it walks at only 50-60 % of the commanded forward
speed, and it does not walk sideways. We checked the measurement against the distance
that the robots walked. This is the behavior of the policy, not an error in the
evaluation. The payload experiments will therefore compare each policy with its own
result without payload.

Video (one robot per command, caption shows command and measured velocity):
[results/video/p0_seed0_dev1000.mp4](results/video/p0_seed0_dev1000.mp4)

## How to run

The training runs in WSL2 (Ubuntu 24.04) with the locked environment of the upstream
repository. Clone upstream into `third_party/microduck_rl`, check out commit `bae39f5`,
and run `uv sync --locked` there. Then run all commands from the
upstream folder:

```bash
bash scripts/train_p0.sh 4096 3000 0
uv run --locked python <this repo>/scripts/evaluate.py --checkpoint <run>/model_2999.pt --seed 1000 --out <folder>
uv run --locked python <this repo>/scripts/record_video.py --checkpoint <run>/model_2999.pt --out <file>.mp4
uv run --locked --with pytest pytest -q <this repo>/tests
```

## Layout

| Path | Content |
|---|---|
| `src/payload_duck/` | Evaluation environment (fixed conditions) and metrics |
| `scripts/` | Training, evaluation, video, GPU monitor, speed test |
| `tests/` | Unit tests for the metrics and the evaluation environment |
| `results/` | Training curves, evaluation summaries, videos |
