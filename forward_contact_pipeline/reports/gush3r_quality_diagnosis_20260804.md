# GUSH3R GoodMornin1 quality diagnosis — 2026-08-04

## Scope and fixed input

This is a rendering/reconstruction diagnosis for GUSH3R, independent of the
forward-contact P0 decision.  All runs use the same Human3R-exported RGB
frames from `forward_contact_pipeline/visualizations/human3r_goodmornin_full/`
at 512x368.  Frames are exported under
`forward_contact_pipeline/diagnostics/gush3r_quality_20260804/`; no input
video is decoded through a system temporary directory.

The original severe output is
`visualizations/gush3r_goodmornin_official_full/`: background white-outs,
black floating dots and progressive blur are visible while the humans remain
relatively plausible.

## Diagnostic code and reproducibility changes

- `GUSH3R/infer.py`
  - added `--save_components`, which emits background-only, externally
    rasterized-human-only, model-native-human and merged render diagnostics;
  - added `--background-render-mode {final,per-frame}`;
  - `per-frame` requests background Gaussian snapshots from the existing
    model inference API and renders frame *t* with its own accumulated
    background, instead of rendering every frame with the last background;
  - this is a diagnosis mode, not a changed checkpoint or training recipe.
- `GUSH3R/src/mhmr/blocks/dinov2.py` supports
  `GUSH3R_DINO_HUB_LOCAL_REPO`: with this variable it calls Torch Hub with
  `source="local"`.  This avoids Hub probing GitHub for the default branch
  even when a local cache exists.
- `forward_contact_pipeline/scripts/export_frameinput_rgb.py` exports fixed
  RGB windows from FrameInput (`--start-frame`, `--max-frames`) entirely under
  the workspace.
- A local DINO hub source cache is placed under this diagnostic directory;
  it makes the experiments independent of intermittent GitHub access.

## Completed evidence

### 1. Component separation, official CLI settings, first 32 frames

Run: `official_cli_32/`, with `gs_conf_threshold=1.0`,
`bg_mask_threshold=0.02`, `bg_mask_dilation=3`, `bg_voxel_size=0.005`,
`bg_gaussian_max=2,000,000`.

- Background-only render is a coherent kitchen scene with intentional holes
  at detected people.
- External human-only render is visually plausible and fills those holes in
  the merged render.
- Model-native `human_rgb` is premultiplied/transparent-on-black, so it must
  not be judged as a standalone RGB image without its alpha compositing.
- The merged 32-frame output is substantially cleaner than the old 229-frame
  output.  Therefore the bad quality is neither an unavoidable checkpoint
  failure nor primarily an external rasterizer failure.

### 2. Window-length ablation, official CLI settings

| Run | Frames | Result |
|---|---:|---|
| `official_cli_32` | 0--31 | coherent background, modest blur |
| `official_cli_64` | 0--63 | still usable, mild wall loss/blur |
| `official_cli_128` | 0--127 | floating dots at scene boundary and stronger smoothing |
| historical official full | 0--228 | severe black dots/white regions/blur |

The fused background reaches the 2,000,000-Gaussian cap early (well before
the end) and remains at that cap.  Since `infer.py` selects the **last**
background Gaussian collection and uses it for every rendered frame, later
fusion errors affect even frame 0.  This explains why the old frame 0 is bad
despite the first 32-frame reconstruction itself being good.

### 3. Background-cap and voxel ablations at 128 frames

| Run | Key settings | Finding |
|---|---|---|
| `official_cli_128` | 2M, voxel .005, conf 1.0, mask .02 | best coverage, but long-window dots/blur |
| `model_defaults_128` | 200k, voxel .01, conf 1.5, mask .05 | much worse large white holes; do not use |
| `official_2m_voxel01_128` | 2M, voxel .01, other official settings | slightly fewer boundary dots/smoother, but not a cure |

The result rules out a simple "reduce Gaussian count" fix.  The model's
constructor defaults and the public infer CLI defaults are materially
different; the CLI 2M/.005 configuration is better for scene coverage.

### 4. Full 229-frame voxel .01 check

Run: `official_2m_voxel01_229/`, 229 frames, 2M cap, voxel .01.
Inference completed in 90.87 seconds.  It remains poor at full length (and
can show more black noise at early frames), confirming that voxel size alone
does not resolve temporal accumulation drift.

### 5. Direct causal-background render test

Run: `causal_bg_32/` with `--background-render-mode per-frame`; comparison
image: `final_vs_causal_32.png`.

This uses the identical 32-frame inference but renders frame *t* with the
background available at *t*.  It yields a visibly more complete and sharper
room background than the standard final-background render.  It also exposes
human-shaped holes because the background mask deliberately excludes people
and human Gaussian compositing is not yet perfect.  This is direct evidence
that final-background back-propagation is a major source of the observed
historical quality collapse.

## Current conclusion

1. Human Gaussian estimation is not the dominant source of the old failure.
2. The main failure is long-horizon background fusion plus rendering every
   historical frame from the final fused background.  This creates future to
   past contamination; the 2M cap and noisy cross-view geometry compound it.
3. `bg_gaussian_max=200k` is not an acceptable remedy because it produces
   large missing-background regions.  `voxel=.01` is at most a minor visual
   trade-off, not a production fix.
4. Do not use the historical 229-frame full render as the baseline for
   contact-quality claims.  It confounds GUSH3R background fusion/rendering
   failure with any future contact correction.

### 6. Late-content fresh-window control

`late_window_32/` completed on original frames 192--223 with the official
2M/.005 settings (13.20 seconds inference).  Comparison image:
`late_window_vs_full.png`.

The same late content is severely blurred/noisy in the 229-frame global
render, but is substantially sharper and more coherent as a fresh 32-frame
window.  Its remaining failure is mainly white holes around/behind people,
which is consistent with the intentionally excluded background human mask.
This independently confirms that the major collapse comes from long-window
global fusion/rendering, not because frames 192--223 are intrinsically too
difficult for the checkpoint.

## Recommended engineering direction after that validation

- Preserve/render causal or bounded-window background states rather than a
  single final background for all video frames.  To avoid retaining hundreds
  of 2M-Gaussian snapshots in GPU memory, implement streaming snapshots or
  chunked inference/rendering (e.g. 32--64-frame windows), then compose the
  human Gaussian with the matching local background.
- Evaluate the resulting chunked/causal renderer before touching contact
  correction, and report background coverage, black/white artifact rate,
  temporal consistency, and the contact ROI separately.

## Implemented bounded-window renderer (2026-08-05)

`GUSH3R/infer.py` now provides a production-oriented
`--background-render-mode chunked` option. It splits a long input sequence
into independent bounded windows (default 32 frames), runs the recurrent
inference entry point separately for each window, renders only with that
window's background collection, writes globally indexed PNGs, and encodes a
single final MP4 without retaining the whole video in RAM. It also writes
`chunked_render_manifest.json` with the exact window boundaries.

Example (the existing diagnostic RGB directory has 229 frames):

```bash
GUSH3R_DINO_HUB_LOCAL_REPO=/absolute/path/to/local/dinov2 \
  /workspace/nas_auto_backup/nas/yuzilang/miniconda3/envs/gush3r/bin/python infer.py \
  --seq_path ../forward_contact_pipeline/diagnostics/gush3r_quality_20260804/input_frames_229 \
  --output_dir ../forward_contact_pipeline/diagnostics/gush3r_quality_20260804/chunked32_229_fixed \
  --size 512 --device cuda \
  --gs_conf_threshold 1.0 --bg_mask_threshold 0.02 --bg_mask_dilation 3 \
  --bg_voxel_size 0.005 --bg_gaussian_max 2000000 \
  --background-render-mode chunked --background-chunk-size 32
```

This is an engineering fix derived from the completed causal/window evidence;
it preserves the checkpoint and the original `final`/`per-frame` modes. Its
full 229-frame visual/temporal evaluation remains pending GPU availability.

## 2026-09-06: completed numeric audit of the 229-frame chunked output

The completed run is
`diagnostics/gush3r_quality_20260804/chunked32_229_verified/`.  Its manifest
records eight independent windows (seven 32-frame windows plus a final
5-frame window), 229/229 rendered PNGs and 303.42 s total elapsed time.

To keep this audit reproducible without visual inspection, the new script
`forward_contact_pipeline/scripts/analyze_gush3r_render_temporal.py` decodes
frames only inside a local numeric process and writes aggregate JSON; it never
produces a contact sheet or emits image content.  The complete result is
`chunked32_229_verified/temporal_audit_vs_official_full.json`; the historical
full-run control is `official_full_temporal_audit.json`.

| Aggregate proxy (all rendered pixels) | Global 229-frame final map | 32-frame independent chunks |
|---|---:|---:|
| Mean pure-white-pixel rate | 13.91% | 6.65% |
| 95th-percentile pure-white-pixel rate | 60.71% | 28.81% |
| Mean pure-black-pixel rate | 0.63% | 0.72% |
| Mean consecutive-frame RGB L1 | 0.06378 | 0.04491 |

However, the chunk boundaries (`32,64,96,128,160,192,224`) are not temporally
continuous: their mean RGB L1 is `0.16833`, versus `0.04100` for the 221
within-window transitions (4.11x); the maximum boundary jump is `0.31569`.
This is expected because every chunk resets both recurrent context and its
background map.  The result is therefore a *validated anti-drift diagnostic*,
not yet a valid online-streaming renderer or contact baseline.

The next reconstruction experiment must retain the bounded horizon while
transferring a reliable overlap anchor across windows (or, as a weaker
render-only control, explicitly blend an overlap).  A simple map freeze is
rejected because it failed to fill later human-occluded background, and a
per-frame full-map Top-K is rejected because it is not online-feasible at the
2M-Gaussian cap.  Any successor must be assessed against both artifact rates
and the boundary-to-interior temporal-L1 ratio, separately from contact
metrics.

## 2026-09-06: automatic seam and causal-context ablations

All results below use the same 229 input frames and only aggregate numeric
image statistics; no visual inspection was used for the decision.

| Variant | Future input at render time | Mean white rate | Boundary L1 | Interior L1 | Boundary / interior | Wall time |
|---|---|---:|---:|---:|---:|---:|
| hard 32-frame chunks | no | 6.65% | 0.16833 | 0.04100 | 4.11x | 303.4 s |
| 32-frame + 16-frame overlap, linear RGB blend | **yes** | 3.13% | 0.04222 | 0.03866 | 1.09x | 560.6 s |
| 16-frame chunks + 8 historical frames + causal snapshots | no | 7.07% | 0.10950 | 0.04742 | 2.31x | see manifest |

The overlap result (`chunked32_overlap16_229_v1/`) is a successful *offline
seam-control*: it reduces boundary L1 by 74.9% relative to hard chunks and
makes boundaries statistically similar to ordinary transitions.  It must not
be described as online, because each blended historical output uses the next
window.

## 2026-09-09: carried-background-anchor smoke test (negative result)

This test uses 64 input frames split into two 32-frame windows and evaluates
only aggregate pixel statistics; no image was inspected or emitted.  The
second window receives up to 1,000,000 top-confidence background Gaussians
exported from the first.  The new `anchor_budget` cap policy reserves their
voxel budget before randomly selecting new candidate voxels, so this is a
direct test of whether random cap eviction was the reason earlier anchor
handoff failed.

| Run | Carried anchors | Cap policy | White rate | Boundary L1 | Interior L1 | Boundary / interior |
|---|---:|---|---:|---:|---:|---:|
| `anchor250k_64_smoke_v1` | 250k | legacy random | 14.65% | 0.24029 | 0.02810 | 8.55x |
| `anchor1m_64_smoke_v1` | 1M | legacy random | 14.66% | 0.23979 | 0.02812 | 8.53x |
| `anchor1m_budget_64_smoke_v1` | 1M | protected anchor budget | 14.68% | 0.24067 | 0.02814 | 8.55x |

`anchor_budget` is therefore a confirmed negative result: protecting the
carried slots/voxels does **not** repair the discontinuity, and the failure is
not explained by the anchor count or simple random-cap eviction.  The newest
variant takes 80.41 seconds, renders all 64 frames, and has no missing output,
so this is a valid functional test rather than an interrupted run.  It must
not be expanded to 229 frames.

The evidence points to a more fundamental state mismatch: independently
recurrently inferred window-two camera/geometry/features are not compatible
with simply injecting a static selection of window-one Gaussian slots.  The
next online candidate must carry an explicitly maintained map state with
per-anchor validity/reprojection support, age/stability and deterministic
replacement, rather than top-confidence export plus slot preservation.  Until
such a state update passes the same boundary test, the causal-context result
(boundary/interior 2.31x) remains the strongest future-free engineering
control, while overlap blending remains offline-only.

The future-free context result (`chunked16_context8_causal_229_v2/`) is also
fully complete (229/229 frames) and bounds its snapshot memory to 24 frames;
the earlier 32+16 context attempt exited after its first window because
32--48 two-million-Gaussian snapshots exceeded the 4090 memory budget.  It
does improve the hard-switch seam by 35.0%, but remains 2.31x above its
interior transitions and slightly worsens white/black artifact proxies.  Thus
RGB history alone does not recreate the missing map state.

**Updated decision.** The required next model change is an explicit,
budgeted *background anchor hand-off*: serialize a selected stable subset of
the preceding map and inject it as the initial map of the next recurrent
window.  It must preserve only high-confidence, non-human, cross-view-stable
Gaussians; reserve capacity for newly exposed (especially formerly
human-occluded) background; and be tested against hard chunk/context/overlap
with the same boundary-L1 and artifact protocol.  Do not claim the current
context CLI is a latency-proven online runtime: it is future-free rendering
semantics but still recomputes finite windows rather than serializing the
backbone recurrent state.
