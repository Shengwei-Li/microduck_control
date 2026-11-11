# Preregistration D2: payload in the beak

Date: 2026-11-11. This file is committed before any run on the test seeds.

## Setting (frozen)

- Robot and task: Microduck, `Mjlab-Velocity-Flat-MicroDuck`, upstream commit `bae39f5`.
- Payload: a ball that is fixed to the tip of the beak (upstream site `mouth_tip`).
  It has no contact with the ground or the robot. The robot mass is 0.737 kg.
- The payload appears 5 s after the start of the episode. Before that, the robot
  walks without payload.
- Payload masses: 0, 20, 40, 60, 80, 100, 120, 150, 200 g.
- Commands: stand, forward_slow, forward_fast, backward, turn, forward_turn
  (values in `src/payload_duck/eval_env.py`). One fixed command for each robot.
- Episode: 20 s. No curriculum, no random pushes. The other upstream domain
  randomization stays on.
- Seeds: training 0-99, development 1000-1999, test 2000-2999.
- Test runs: seeds 2000 and 2001, 64 robots per command and mass for each seed.
  This gives 128 episodes per command and mass, and 768 episodes per mass.

## Metrics

- Fall rate: fraction of episodes that end with the upstream `fell_over` termination.
- Mean velocity and tracking RMSE: from 7 s (2 s after the payload appears) to the
  end of the episode or to the fall. If more than half of the episodes of a group
  fall, we report the tracking values but do not use them in a hypothesis.
- Time to fall: fall time minus 5 s.

## What we already know

All hypotheses below come from development seed 1000 (384 episodes per mass),
with the baseline policy P0 (seed 0, no payload in training):

| Mass (g) | 0 | 20 | 40 | 60 | 80 | 100 | 120 | 150 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| Fall rate | 0 | 0 | 0 | 0.003 | 0.023 | 0.156 | 0.372 | 0.745 | 0.992 |
| forward_fast vx (m/s), command 0.4 | 0.22 | 0.26 | 0.30 | 0.33 | 0.35 | 0.33 | 0.30 | 0.27 | 0.12 |
| backward vx (m/s), command -0.3 | -0.19 | -0.14 | -0.08 | -0.04 | 0.00 | 0.03 | 0.04 | 0.05 | 0.03 |

The payload also pushes the head down, because the neck servos are soft.

## H1: failure of the baseline policy P0 (test in D2)

H1 uses the same policy as the development result. It is a check that the result
holds on new seeds, and it fixes the numbers that the other methods must beat.

- H1a. The fall rate over all commands is at most 0.02 for each mass up to 60 g.
- H1b. The fall rate over all commands goes above 0.5 between 120 g and 150 g:
  it is below 0.5 at 120 g and above 0.5 at 150 g.
- H1c. Tracking fails before balance. At 60 g, the mean backward velocity is less
  than half of its value at 0 g (in magnitude), while the fall rate at 60 g is at most 0.02.
- H1d. Forward walking is the most fragile. At 100 g and at 120 g, forward_fast has
  the highest fall rate of the six commands, and backward has the lowest.

## H2-H4: methods (test in D3)

Methods: P1 trains with a random payload mass (uniform, 0-100 g). P2 is P1 plus the
true payload mass in the observation. P3 gives the actor a history of observations,
and gives the critic the true payload mass. Each method has 3 training seeds. We
report the median over the 3 seeds. Masses of 120 g and more are outside the
training range. Training details that are not listed here are chosen on the
development seeds and then frozen.

- H2a. P1 is robust inside its training range: its fall rate is at most 0.05 for
  each mass up to 100 g.
- H2b. Robustness has a cost without payload: at 0 g, the tracking RMSE (xy) of P1 is
  at least 5 % higher than that of a policy trained in the same way without payload.
- H3a. Inside the training range (mean over 20-100 g), P3 closes at least half of
  the gap in tracking RMSE (xy) between P1 and P2.
- H3b. Outside the training range (150 g), P3 has a lower fall rate than P1.
- H4. The payload mass can be read from the last hidden layer of the P3 actor with
  a linear model: R^2 > 0.7 on episodes that were not used to fit the model.

The hypotheses for the runtime monitors (D4) will be in a separate file. It will be
committed before the monitor tests. It will include a simple baseline: a threshold
on the position error of the neck joint.
