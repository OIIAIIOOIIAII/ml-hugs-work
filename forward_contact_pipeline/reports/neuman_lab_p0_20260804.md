# NeuMan lab P0 calibration report — 2026-08-04

## Scope

This is a coordinate/proxy calibration report, not a contact-GT evaluation.
The reference is NeuMan `lab`: its aligned 4D-Humans SMPL and COLMAP scene
define the HUGS/NeuMan scene world. Human3R was run on the same 103 RGB frames.

## Reproducible artifacts

- Human3R FrameInput: `../datasets/neuman_lab_frameinput_p0_v2/`
  - 103 frames, Human3R 896L, 512-pixel input.
  - Model inference time reported by the exporter: 33.63 s.
- Approximate Human3R-to-HUGS conversion:
  `../datasets/neuman_lab_h3r_hugs_p0/`.
- SMPL-X-to-SMPL fitted conversion:
  `../datasets/neuman_lab_h3r_hugs_fitted_p0_py38/`.
  - Internal fit: vertex L1 0.008333 m, joint L1 0.011536 m, bbox L1 5.953 px.
- Body-vertex correspondence reports:
  `../datasets/neuman_lab_p0/correspondences.json` and
  `correspondences_fitted.json`.
- Camera-trajectory alignment diagnostic:
  `../datasets/neuman_lab_p0/camera_aligned_scene_points.json`.
- Prepared proxy sequences:
  `../prepared/neuman_lab_p0_scaledproxy.npz` and
  `neuman_lab_p0_fittedproxy.npz`.

## Alignment evidence

| Reference used for Sim(3) | Scale | Hold-out / camera median error (NeuMan world units) |
|---|---:|---:|
| same-frame approximate SMPL vertices | 9.0266 | 1.3139 |
| same-frame fitted SMPL vertices | 9.1028 | 1.4070 |
| same-frame camera centres | 9.6989 | 0.9639 |

The fitted body representation improves Human3R-internal SMPL fidelity but
does **not** improve cross-method NeuMan alignment. This means the dominant
error is the independently reconstructed Human3R-versus-NeuMan pose/camera
geometry, not the SMPL-X-to-SMPL topology conversion.

## Proxy result and decision

The initial real run revealed an implementation bug: anchors were transformed
to NeuMan world but the local-proxy radius was still in native Human3R units.
`prepare_sequence.py` now maps both point map and radius with the same Sim(3).
With that fix:

| Transform | Proxy-confidence median | Frames/feet with confidence >= 0.5 | Absolute surface-distance median | Tracking frames | Rule contact rate |
|---|---:|---:|---:|---:|---:|
| approximate-body Sim(3) | 0.2607 | 29.1% | 3.7691 | 20 / 103 | 0% / 0% |
| fitted-body Sim(3) | 0.2607 | 29.1% | 3.8009 | 21 / 103 | 0% / 0% |

**Decision: P0 software and data-generation are complete, but P0 geometry
acceptance is not met.** The result must not be used as contact labels or to
train the GRU/TCN. The `Uncertain` fallback is behaving as intended.

## Next P0 work

1. Make camera/scene Sim(3) a first-class correspondence candidate rather
   than relying only on independently estimated body vertices.
2. Add per-frame mesh/mask reprojection and 3-D anchor/proxy visualizations;
   use them to locate whether the dominant failure is camera pose, human scale,
   scene point map, or local-radius/normal convention.
3. Calibrate the reference proxy against NeuMan COLMAP scene points/depth
   before treating signed distance as a physical contact measurement.
4. Only after the above diagnostics pass, implement P1 IK/projection and use
   BEDLAM/PROX for true contact supervision.
