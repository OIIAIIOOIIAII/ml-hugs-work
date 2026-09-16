# Stage-A proximity uncertainty / abstention experiments

**Result class:** oracle local-relation proxy robustness; no RGB token, no
forward Gaussian input, no controller.

## Protocol and gate

The local encoder predicts signed SDF proximity and a differentiably bounded
Laplace residual scale.  Low scale is the hypothetical “execute correction”
set.  The gate is valid only if (a) retained low-scale samples have lower MAE
than all samples and (b) error increases with uncertainty bins.

## v1 — invalid implementation

Hard `exp(logit).clamp` saturated every scale at 50 mm, killing the scale-head
gradient.  This artifact is retained but has no result claim.

## v2 — strict unseen-error-family test

Train uses clean + normal18; validation uses dropout20; test is unseen
distance corruption, with scene/sequence/error-family separated.

- Validation bins are monotonic: low to high scale MAE = 0.123, 0.064, 0.722,
  1.026, 1.268 mm.
- Test all-sample MAE = 0.698 mm; keeping the predicted lowest-scale 50% gives
  **0.743 mm**, worse than all samples.

**Verdict:** no unseen-error OOD reliability.  Do not connect this head to
Stage B.

## v3 / augmented-scene protocol

Every declared proxy error family (clean, normal18, dropout20, distance) is
available during training; scenes remain disjoint.  This is not an OOD result.

- Test all-sample MAE = 0.716 mm.
- Keep lowest scale 25/50/75%: 0.768 / **0.653** / 0.707 mm.
- Test five scale bins have MAE = 0.858, 0.479, 0.640, 0.913, 0.689 mm: not
  monotonic.

The 50% figure alone is insufficient: the 25% result and bin ordering show
that the model has no stable calibrated abstention signal.  The small local
point/mesh relation head is therefore inadequate for a safety gate.  The next
valid route is to add frozen RGB/Gaussian relation tokens and train against a
measured coarse-front-end error/reliability target, then repeat calibration;
do not continue tuning this head or advance to the causal controller.

Artifacts:

- `runs/stagea_proximity_uncertainty_v2_20260909/model/metrics.json`
- `runs/stagea_proximity_uncertainty_aug_v1_20260909/model/metrics.json`
