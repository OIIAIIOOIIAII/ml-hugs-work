# Human3R Project Intro for AI Agents

This document is a compact project description intended to be copied into other
repositories and read by AI coding/research agents. It focuses on what Human3R
does, how to call it, what inputs it accepts, and the exact output data formats.

## 1. Project Summary

Human3R is an inference/training codebase for online human-scene 3D
reconstruction from monocular RGB video or image sequences.

At inference time, the main demo runs one Human3R checkpoint on a sequence and
predicts:

- dense per-frame scene geometry as depth / point clouds
- per-frame camera pose and intrinsics
- per-frame SMPL-X human parameters for all detected people
- optional visualization images and an interactive Viser viewer

The primary inference entry point is:

```bash
python demo.py
```

The repository also contains:

- `inference_only.py`: forward-pass timing script
- `scripts/export_ply_from_saved_results.py`: converts saved Human3R frame
  outputs into colored `.ply` point clouds
- `docs/inference.md`: checkpoint variants and speed/accuracy notes
- `docs/eval.md`: evaluation usage
- `docs/train.md`: training usage

## 2. Required Runtime Assets

Human3R inference expects model assets under `src/`.

Typical checkpoint:

```text
src/human3r_896L.pth
```

SMPL/SMPL-X assets are also expected under:

```text
src/models/smpl/
src/models/smplx/
src/models/smpl_mean_params.npz
src/models/smpl_colors.txt
```

Checkpoint choices documented by the project include:

- `human3r_672S.pth`: faster, lower detail
- `human3r_672B.pth`
- `human3r_672L.pth`
- `human3r_896L.pth`: slower, highest reported detail

## 3. Main Inference Command

Example command for saving results without launching the viewer:

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --model_path src/human3r_896L.pth \
  --size 512 \
  --seq_path /path/to/video_or_image_folder \
  --output_dir output/my_human3r_run \
  --subsample 1 \
  --use_ttt3r \
  --vis_threshold 2 \
  --downsample_factor 1 \
  --reset_interval 100 \
  --save \
  --no_viewer
```

Useful arguments:

| Argument | Meaning |
| --- | --- |
| `--model_path` | Path to a Human3R `.pth` checkpoint. |
| `--seq_path` | Input video file or directory of image frames. |
| `--output_dir` | Directory where saved outputs are written when `--save` is set. |
| `--save` | Enables writing output files to disk. |
| `--no_viewer` | Skips interactive Viser viewer; useful for batch processing. |
| `--render` | Saves SMPL projection visualization frames to `color_smpl/`. |
| `--render_video` | With `--render`, also saves `output_video.mp4`. |
| `--size` | Image resize target passed to the model, typically `512`. |
| `--max_frames` | Optional cap on number of input frames. |
| `--subsample` | Uses every N-th input frame. |
| `--use_ttt3r` | Enables TTT3R recurrent inference option. |
| `--reset_interval` | Interval for recurrent state reset. |
| `--device` | `cuda` or `cpu`; default is `cuda`. |

## 4. Input Format

`--seq_path` accepts either a video file or an image folder.

### 4.1 Video Input

Supported through OpenCV `VideoCapture`, for example:

```text
examples/GoodMornin1.mp4
examples/yogaball.mp4
```

Human3R extracts all frames internally into a temporary directory, then applies
`--max_frames` and `--subsample`.

### 4.2 Image Folder Input

If `--seq_path` is a directory, all files directly inside that directory are
loaded after lexicographic sorting:

```text
/path/to/sequence/
├── 000000.png
├── 000001.png
├── 000002.png
└── ...
```

Use zero-padded filenames so sorting matches temporal order.

### 4.3 Image Tensor Convention Inside the Model

Images are loaded by `src.dust3r.utils.image.load_images`.
The model receives normalized tensors, and saved colors are converted back to
standard RGB PNG files in `[0, 255]`.

## 5. Saved Output Directory

When `demo.py` is run with `--save`, the output directory has this structure:

```text
output/my_human3r_run/
├── color/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── depth/
│   ├── 000000.npy
│   ├── 000001.npy
│   └── ...
├── conf/
│   ├── 000000.npy
│   ├── 000001.npy
│   └── ...
├── camera/
│   ├── 000000.npz
│   ├── 000001.npz
│   └── ...
├── smpl/
│   ├── 000000.npz
│   ├── 000001.npz
│   └── ...
├── color_smpl/          # only if --render is used
│   ├── 000000.png
│   └── ...
├── output_video.mp4     # only if --render --render_video are used
└── ply_export/          # only if export_ply_from_saved_results.py is run
    ├── 000000.ply
    ├── ...
    └── merged.ply       # only if --merge is used
```

Frame stems are zero-padded six-digit indices (`000000`, `000001`, ...). These
indices refer to the processed sequence after `--max_frames` and `--subsample`,
not necessarily the original video frame numbers.

## 6. Output Data Formats

The following shapes are per processed frame. Let:

- `H, W`: saved frame resolution after model preprocessing/output
- `N`: number of detected humans in the frame

An observed run produced `H=368, W=512`, but consumers should read array shapes
dynamically instead of hard-coding these values.

### 6.1 `color/<frame>.png`

RGB image used for the saved frame.

Format:

```text
PNG, uint8, shape (H, W, 3), RGB, value range [0, 255]
```

### 6.2 `depth/<frame>.npy`

Dense predicted depth map in the current camera coordinate system.

Format:

```text
NumPy .npy, float32, shape (H, W)
```

Usage:

- `depth[y, x]` is the predicted positive z-depth for pixel `(x, y)`.
- Invalid or unwanted pixels should be filtered with finite checks, positive
  depth checks, and confidence thresholding.

### 6.3 `conf/<frame>.npy`

Dense confidence map for the predicted scene depth/points.

Format:

```text
NumPy .npy, float32, shape (H, W)
```

Usage:

- Higher values indicate higher confidence.
- The PLY export script defaults to `conf_threshold=1.5`.

### 6.4 `camera/<frame>.npz`

Camera parameters for the frame.

Keys:

```text
pose        float32, shape (4, 4)
intrinsics  float32, shape (3, 3)
```

`pose` is camera-to-world (`c2w`).

`intrinsics` is a standard pinhole matrix:

```text
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```

Backproject a pixel to world coordinates:

```python
z = depth[y, x]
x_cam = (x - cx) * z / fx
y_cam = (y - cy) * z / fy
p_cam = [x_cam, y_cam, z]
p_world = p_cam @ pose[:3, :3].T + pose[:3, 3]
```

This is the same convention used by
`scripts/export_ply_from_saved_results.py`.

### 6.5 `smpl/<frame>.npz`

SMPL-X human prediction data for the frame.

Keys:

```text
scores      float32, shape (H, W)
msk         float32, shape (1, H, W), or None when mask output is absent
shape       float32, shape (N, 10)
rotvec      float32, shape (N, 53, 3)
transl      float32, shape (N, 3)
expression  float32, shape (N, 10), or None when expression output is absent
```

Meanings:

- `N` is the number of humans predicted in this frame.
- `shape` contains SMPL-X beta parameters.
- `rotvec` contains axis-angle pose vectors for 53 SMPL-X joints/parts.
- `transl` contains per-human translation.
- `expression` contains SMPL-X expression coefficients when available.
- `scores` is a dense human/SMPL score map saved at image resolution.
- `msk` is a dense mask-like output when the model provides one.

To reconstruct meshes from these parameters, use the repository's
`SMPL_Layer(type="smplx", gender="neutral", person_center="head")` logic in
`demo.py::prepare_output`. That function also transforms SMPL vertices into the
world frame using the per-frame camera pose.

### 6.6 `color_smpl/<frame>.png` and `output_video.mp4`

These are visualization-only outputs produced by `--render` and
`--render_video`.

`color_smpl/<frame>.png` horizontally concatenates:

- RGB input/output color
- optional mask heatmap
- SMPL score heatmap
- rendered SMPL mesh overlay

Do not treat this image as a geometry source; use `depth`, `camera`, and `smpl`
for structured data.

## 7. PLY Export Format

After running Human3R with `--save`, export colored point clouds with:

```bash
python scripts/export_ply_from_saved_results.py \
  --result_dir output/my_human3r_run \
  --frames all \
  --conf_threshold 1.5 \
  --stride 1 \
  --merge
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `--result_dir` | Saved Human3R result directory containing `color/depth/conf/camera`. |
| `--frames` | `all`, comma list such as `0,10,25`, or ranges such as `0-30`. |
| `--out_dir` | Optional output directory; default is `<result_dir>/ply_export`. |
| `--conf_threshold` | Keeps points with confidence greater than or equal to threshold. |
| `--stride` | Spatial pixel stride for lighter point clouds. |
| `--merge` | Also writes one merged cloud as `merged.ply`. |

Output:

```text
ply_export/<frame>.ply
ply_export/merged.ply
```

Each `.ply` is a colored point cloud created by backprojecting valid depth
pixels into world coordinates with the saved camera intrinsics and `c2w` pose.

Conceptual point fields:

```text
x, y, z      float world coordinates
red, green, blue uint8 RGB color
```

Validity filter used by the export script:

```python
valid = np.isfinite(depth) & np.isfinite(conf) & (depth > 0) & (conf >= conf_threshold)
```

## 8. Coordinate and Indexing Notes

- Pixel coordinates use image convention: `x` is column, `y` is row.
- Depth is camera-space positive z.
- `camera/<frame>.npz::pose` is camera-to-world.
- Exported PLY points are in the Human3R predicted world coordinate system.
- Saved frame IDs are processed-frame IDs after video extraction, frame cap, and
  subsampling.
- For videos, original frame numbers are not preserved in saved filenames by
  default.

## 9. Minimal Consumer Recipes

### 9.1 Load One Frame's Structured Output

```python
from pathlib import Path
import numpy as np
from PIL import Image

result_dir = Path("output/my_human3r_run")
frame = "000000"

color = np.array(Image.open(result_dir / "color" / f"{frame}.png"))
depth = np.load(result_dir / "depth" / f"{frame}.npy")
conf = np.load(result_dir / "conf" / f"{frame}.npy")
cam = np.load(result_dir / "camera" / f"{frame}.npz")
smpl = np.load(result_dir / "smpl" / f"{frame}.npz", allow_pickle=True)

pose_c2w = cam["pose"]
K = cam["intrinsics"]
shape = smpl["shape"]
rotvec = smpl["rotvec"]
transl = smpl["transl"]
```

### 9.2 Build a Point Cloud Array From One Frame

```python
import numpy as np

H, W = depth.shape
yy, xx = np.indices((H, W), dtype=np.float32)

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

z = depth.astype(np.float32)
x = (xx - cx) * z / fx
y = (yy - cy) * z / fy
points_cam = np.stack([x, y, z], axis=-1)

R = pose_c2w[:3, :3].astype(np.float32)
t = pose_c2w[:3, 3].astype(np.float32)
points_world = points_cam @ R.T + t

valid = np.isfinite(depth) & np.isfinite(conf) & (depth > 0) & (conf >= 1.5)
points = points_world[valid]
colors = color[valid]
```

## 10. Common Integration Guidance

For another project consuming Human3R output:

- Use `depth + camera` for dense scene geometry.
- Use `conf` to filter unreliable points.
- Use `smpl` for per-person SMPL-X parameters.
- Use `color` for RGB point colors or visual alignment.
- Use `ply_export` when the downstream system only needs colored point clouds.
- Avoid parsing rendered visualization files as data.
- Do not assume a fixed resolution or fixed number of humans per frame.

