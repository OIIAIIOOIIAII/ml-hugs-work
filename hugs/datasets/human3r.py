#
# Human3R converted-output dataset for HUGS.
#

import glob
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from hugs.utils.graphics import BasicPointCloud, get_projection_matrix


def get_center_and_diag(cam_centers):
    cam_centers = np.vstack(cam_centers)
    center = np.mean(cam_centers, axis=0, keepdims=True)
    dist = np.linalg.norm(cam_centers - center, axis=1, keepdims=True)
    return center.flatten(), np.max(dist)


def get_data_splits(num_frames):
    if num_frames < 3:
        return list(range(num_frames)), list(range(num_frames)), []

    val_list = list(range(2, num_frames, 5))
    if len(val_list) == 0:
        val_list = [num_frames - 1]
    train_list = [i for i in range(num_frames) if i not in val_list]
    if len(train_list) == 0:
        train_list = list(range(num_frames))
    return train_list, val_list, []


class Human3RDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, render_mode="human_scene", cache_to_cuda=True):
        self.root = root
        self.split = split
        self.mode = render_mode

        if not root:
            raise ValueError("Human3RDataset requires cfg.dataset_path to point at a converted Human3R dataset")
        if not os.path.isdir(root):
            raise FileNotFoundError(root)

        self.image_paths = sorted(glob.glob(os.path.join(root, "images", "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(root, "masks", "*.png")))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found under {os.path.join(root, 'images')}")
        if len(self.mask_paths) != len(self.image_paths):
            raise RuntimeError("Human3R converted dataset must have one mask PNG per image")

        cameras_path = os.path.join(root, "cameras.npz")
        smpl_path = os.path.join(root, "4d_humans", "smpl_optimized_aligned_scale.npz")
        scene_points_path = os.path.join(root, "scene", "points.npz")
        if not os.path.isfile(cameras_path):
            raise FileNotFoundError(cameras_path)
        if not os.path.isfile(smpl_path):
            raise FileNotFoundError(smpl_path)
        if not os.path.isfile(scene_points_path):
            raise FileNotFoundError(scene_points_path)

        cameras = np.load(cameras_path, allow_pickle=True)
        self.c2w = cameras["c2w"].astype(np.float32)
        self.intrinsics = cameras["intrinsics"].astype(np.float32)
        self.image_sizes = cameras["image_sizes"].astype(np.int32)

        smpl_params = np.load(smpl_path)
        self.smpl_params = {k: torch.from_numpy(smpl_params[k]).float() for k in smpl_params.files}

        if len(self.image_paths) != self.c2w.shape[0]:
            raise RuntimeError("Number of images and cameras does not match")
        if len(self.image_paths) != self.smpl_params["global_orient"].shape[0]:
            raise RuntimeError("Number of images and SMPL frames does not match")

        scene_points = np.load(scene_points_path)
        pcd_xyz = scene_points["points"].astype(np.float32)
        pcd_col = scene_points["colors"].astype(np.float32)
        if pcd_xyz.shape[0] == 0:
            raise RuntimeError("Converted Human3R scene point cloud is empty")
        self.init_pcd = BasicPointCloud(
            points=pcd_xyz,
            colors=np.clip(pcd_col, 0.0, 1.0),
            normals=np.zeros_like(pcd_xyz),
            faces=None,
        )

        _, diag = get_center_and_diag([self.c2w[i, :3, 3] for i in range(self.c2w.shape[0])])
        self.radius = float(max(diag * 1.1, 1e-3))

        self.train_split, self.val_split, _ = get_data_splits(len(self.image_paths))
        self.num_frames = len(self.image_paths)

        self.cached_data = None
        if cache_to_cuda:
            self.load_data_to_cuda()

    def __len__(self):
        if self.split == "train":
            return len(self.train_split)
        if self.split == "val":
            return len(self.val_split)
        if self.split == "all":
            return self.num_frames
        if self.split == "anim":
            return 0
        raise ValueError(self.split)

    def _frame_index(self, i):
        if self.split == "train":
            return self.train_split[i]
        if self.split == "val":
            return self.val_split[i]
        if self.split == "all":
            return i
        if self.split == "anim":
            return i
        raise ValueError(self.split)

    def get_single_item(self, i):
        idx = self._frame_index(i)

        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(self.mask_paths[idx])
        mask = (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)

        rows = np.any(mask > 0.5, axis=0)
        cols = np.any(mask > 0.5, axis=1)
        if rows.any() and cols.any():
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        else:
            h, w = mask.shape
            bbox = np.array([0, 0, h - 1, w - 1], dtype=np.float32)

        h, w = img.shape[:2]
        k = self.intrinsics[idx]
        c2w = self.c2w[idx]
        w2c = np.linalg.inv(c2w).astype(np.float32)

        fovx = 2 * np.arctan(w / (2 * k[0, 0]))
        fovy = 2 * np.arctan(h / (2 * k[1, 1]))
        zfar = 100.0
        znear = 0.01

        world_view_transform = torch.from_numpy(w2c).T
        c2w_t = torch.from_numpy(c2w)
        projection_matrix = get_projection_matrix(
            znear=znear,
            zfar=zfar,
            fovX=fovx,
            fovY=fovy,
        ).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        datum = {
            "rgb": torch.from_numpy(img.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(mask).float(),
            "bbox": torch.from_numpy(bbox).float(),
            "fovx": float(fovx),
            "fovy": float(fovy),
            "image_height": int(h),
            "image_width": int(w),
            "world_view_transform": world_view_transform.float(),
            "c2w": c2w_t.float(),
            "full_proj_transform": full_proj_transform.float(),
            "camera_center": camera_center.float(),
            "cam_intrinsics": torch.from_numpy(k).float(),
            "betas": self.smpl_params["betas"][idx],
            "global_orient": self.smpl_params["global_orient"][idx],
            "body_pose": self.smpl_params["body_pose"][idx],
            "transl": self.smpl_params["transl"][idx],
            "smpl_scale": self.smpl_params["scale"][idx],
            "near": znear,
            "far": zfar,
        }
        return datum

    def load_data_to_cuda(self):
        self.cached_data = []
        for i in tqdm(range(self.__len__()), desc=f"Loading Human3R {self.split} data"):
            datum = self.get_single_item(i)
            for k, v in datum.items():
                if isinstance(v, torch.Tensor):
                    datum[k] = v.to("cuda")
            self.cached_data.append(datum)

    def __getitem__(self, idx):
        if self.cached_data is None:
            return self.get_single_item(idx)
        return self.cached_data[idx]
