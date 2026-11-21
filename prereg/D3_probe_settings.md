# Settings for the test of H4 (linear probe)

Date: 2026-11-21. This file is committed before any probe run on the test seeds.

H4 is in `D2_hypotheses.md`: the payload mass can be read from the last hidden
layer of the P3 actor with a linear model, R^2 > 0.7 on episodes that were not
used to fit the model. That file does not give the details. We chose them on
the development seeds. They are frozen here.

## Settings (frozen)

- Episodes: the standard evaluation of `D2_hypotheses.md` (payload appears at 5 s,
  6 commands, 64 robots per command and mass).
- Input: the 128 outputs of the last hidden layer of the actor (after the ELU),
  checkpoint `model_3998.pt`.
- Target: the payload mass in grams.
- Masses: 0, 20, 40, 60, 80, 100 g (the training range). Larger masses are not used.
- Samples: one robot at one time, every 0.5 s, from 7 s (2 s after the payload
  appears) to the end. Robots that have fallen are not used.
- Model: least squares with intercept. No regularization, no selection of inputs.
- Fit: development seed 1000. Score: test seeds 2000 and 2001, pooled.
- Verdict: median R^2 over the 3 training seeds of P3. H4 holds if it is above 0.7.

## Comparison (not a hypothesis)

We run the same probe on the 3 training seeds of P1. The P1 actor has no
history and no mass input. This shows how much of the result comes from the
history of P3.

## Exploratory

R^2 of the fitted model at each sample time, before and after the payload appears.

## What we already know

Development only: P3 training seed 0, fit on seed 1000, score on seed 1001.
R^2 = 0.86, RMSE 12.8 g. Before the payload appears, R^2 is negative, as it must be.
No run on the test seeds was made before this file.
