# Stage-A continuous proximity + ROI-contact baseline

**Date:** 2026-09-09  
**Result class:** `oracle local-relation mechanism result` (not a real RGB/front-end/Gaussian result)

## Question

After rejecting PROX's pseudo-dense binary foot labels, can a small local
mesh--scene relation encoder recover the valid continuous target, signed SDF
proximity, from a corrupted local scene patch?  Contact is evaluated only at
the whole-foot ROI level.

## Protocol

- Assets: 12 PROX sequences, 60 frames each, PROXD SMPL-X plus official scene
  SDF/mesh; 64 fixed foot/ankle vertices and 128 local scene points per vertex.
- Input: `roi vertex + normal + local point patch + point normal + validity`.
  No RGB is read, no frozen visual token is present, and no Gaussian is used.
- Supervision: per-vertex signed SDF proximity and the existing per-foot
  teacher contact.  The latter is deliberately pooled to one ROI logit.
- Split: existing strict scene/sequence/error-family split; train 1,920,
  validation 240, test 240 foot frames. Train uses clean + normal18 evidence,
  validation dropout20, test distance corruption.
- Model: existing 96D PointNet-like `LocalRelationEncoder`; 40 epochs;
  select the checkpoint by validation signed proximity MAE.

## Result

| Metric | Validation (epoch 27) | Test |
|---|---:|---:|
| Signed proximity MAE | 1.55 mm | **1.27 mm** |
| Absolute proximity MAE | 1.53 mm | **1.05 mm** |
| Proximity RMSE | 2.08 mm | 1.77 mm |
| ROI contact F1 | 0.385 | 0.600 |
| ROI contact precision / recall | 0.433 / 0.347 | 0.429 / 1.000 |

The test ROI classifier again collapses to all-positive.  It is **not** a
valid contact-classification result and must not be used for an F1 claim.

For a simple numerical check, a constant-zero signed-proximity predictor has
test MAE 1.79 mm; the learned local relation encoder reaches 1.27 mm.  A
naive nearest input-point absolute-distance estimator has 12.64 mm MAE on the
distance-corrupted test.  Thus the continuous value contains learnable local
relation signal, but this is still an oracle geometry mechanism result: the
patches are derived from the official scene mesh, not a frozen Gaussian
backbone.

## Decision

Keep continuous proximity as the PROX Stage-A geometry target.  Do not retain
the current ROI contact head as an evidence of reliable contact; next add an
explicit reliability target and calibration/abstention evaluation, then add
frozen RGB/Gaussian tokens only after their real-front-end extraction contract
is implemented.  No Stage-A+B controller or Gaussian LBS claim follows from
this experiment.

Raw artifact: `runs/stagea_proximity_roi_v2_20260909/model/metrics.json`.
