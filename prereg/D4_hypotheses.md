# Preregistration D4: runtime monitors and fallback

Date: 2026-11-24. This file is committed before any monitor run on the test seeds.

The robot, the payload, the commands, and the seeds are those of `D2_hypotheses.md`.

## Question

A policy is safe when the payload is inside its training range (0 to 100 g).
Can a monitor tell, from what the robot measures, that the payload is too
heavy, early enough to prevent the fall?

## Setting (frozen)

- Policies: P1 and P3, 3 training seeds each, checkpoint `model_3998.pt`.
  We report the median over the 3 training seeds, for each method.
- Normal: an episode with a payload from 0 to 100 g. A monitor must be silent
  there. Not normal: 120, 150, 200 g.
- Signals, one value for each robot and control step (50 Hz):
  - neck: position error of the neck joint (target minus measured). The simple baseline.
  - residual: a learned one-step model predicts the next measurement from the
    current measurement and action. The signal is the size of its error.
    The model is trained on normal episodes of development seed 1000.
  - value: minus the output of the critic. The critic reads simulator state
    (base velocity, foot contacts; for P3 also the true payload mass).
    A real robot does not have this signal. It is a reference.
  - probe: the payload mass from the linear probe of H4.
- Monitor: standardize the signal (median and MAD of normal episodes, seed 1000,
  from the time when the payload appears). Alarm when the value is above a
  threshold. No accumulation.
- Threshold: split conformal, from the largest value of each normal episode of
  development seed 1001 (2304 episodes). Target false-alarm rate: 0.05 per episode.
- The monitors are active from 3 s to 13 s of the episode. The payload appears at 5 s.
- Fallbacks, at the first alarm of a robot:
  - stand: the velocity command goes to zero, to the end of the episode.
  - drop: the robot lets go of the payload (its mass goes to zero). There is
    no delay. This is the best case for a release.
- The scales, thresholds, models, and probe weights are in `results/d4/frozen/`.
  They are committed with this file and are not changed after it.
- Test: seeds 2000 and 2001, pooled. 64 robots per command and mass for each seed.

## Metrics

- False-alarm rate: fraction of normal episodes with an alarm.
- Caught: fraction of the falls that have an alarm before the fall.
- Lead time: time from the alarm to the fall, median over the caught falls.
- Delay: time from the payload to the alarm, median over the episodes with an alarm.
- Fall rate with and without a fallback.

## What we already know

Development seed 1002. Median over 3 training seeds (P1 / P3).

| At 200 g | neck | residual | value | probe |
|---|---|---|---|---|
| False-alarm rate (normal episodes) | 0.052 / 0.057 | 0.052 / 0.050 | 0.049 / 0.053 | 0.052 / 0.053 |
| Caught | 0.77 / 0.74 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.96 |
| Lead time (s) | 0.54 / 0.54 | 0.72 / 0.84 | 0.82 / 1.08 | 0.68 / 0.81 |
| Delay (s) | 0.60 / 0.66 | 0.32 / 0.30 | 0.20 / 0.02 | 0.32 / 0.30 |

| Fall rate | no fallback | residual, stand | stand when the payload appears | neck, drop | residual, drop |
|---|---|---|---|---|---|
| P1, 150 g | 0.039 | 0.047 | 0.023 | 0.023 | 0.003 |
| P1, 200 g | 0.440 | 0.372 | 0.281 | 0.258 | 0.029 |
| P3, 150 g | 0.120 | 0.094 | 0.078 | 0.214 | 0.060 |
| P3, 200 g | 0.620 | 0.534 | 0.534 | 0.586 | 0.055 |

A CUSUM on the same signals did not help (allowance 0.5, 1.5, 3.0): the change
is sudden and large. The false alarms are not spread evenly: at 100 g, 21 % to
28 % of the episodes have an alarm; below 60 g, almost none.

The first plan (project specification) said: "at least one monitor has a median
lead time above 0.3 s, and the fallback halves the falls", with stand as the
fallback. On the development seed, stand does not halve the falls, also when
the robot stands from the moment the payload appears. We added the drop fallback
for this reason. H5d below states the result for stand as we now expect it.

## Hypotheses

Each hypothesis is for the median over the 3 training seeds. It must hold for
P1 and for P3, unless it names one method.

- H5a. The calibration holds on new seeds: the false-alarm rate of each of the
  four monitors is at most 0.08.
- H5b. The residual monitor catches the falls at 200 g: caught is at least
  0.95, and the lead time is above 0.3 s.
- H5c. The simple baseline is weaker: at 200 g, caught of the neck monitor is
  lower than caught of the residual monitor by at least 0.10.
- H5d. Stand is not enough: at 200 g, the fall rate with the residual monitor
  and stand is more than half of the fall rate without fallback.
- H5e. Drop works when the alarm is early: at 200 g, the fall rate with the
  residual monitor and drop is at most one quarter of the fall rate without fallback.
- H5f. A late release hurts the adaptive policy: for P3 at 150 g, the fall rate
  with the neck monitor and drop is higher than the fall rate without fallback.
- H6a. For P1 at 200 g, the value monitor is faster than the residual monitor
  (lower delay).
- H6b. For P3 at 200 g, caught of the probe monitor is lower than caught of the
  residual monitor.

## Reported without a hypothesis

- The alarm rate for each mass (the cost: how often a robot stops or drops when
  it would not fall).
- The same monitors with a CUSUM (allowance 3.0).
- Results at 120 g and 150 g for all monitors.

## Limits

- Simulation only. One kind of fault: a payload that is too heavy and appears suddenly.
- The evaluation is not exactly repeatable. Differences of fall rates below
  about 0.03 have no meaning.
- The drop fallback has no delay and no model of the release.
