#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import json
import os
import sys
import time

from loguru import logger
from omegaconf import OmegaConf

sys.path.append(".")

import hugs.trainer.gs_trainer as gst
from hugs.cfg.config import cfg as default_cfg
from hugs.utils.config import get_cfg_items
from hugs.utils.general import safe_state


def make_logdir(cfg, tag):
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    cfg.logdir = os.path.join(
        cfg.output_path,
        cfg.mode,
        cfg.dataset.name,
        cfg.dataset.seq,
        cfg.human.name,
        tag,
        time_str,
    )
    cfg.logdir_ckpt = os.path.join(cfg.logdir, "ckpt")
    for subdir in ("", "ckpt", "val", "train", "anim", "meshes"):
        os.makedirs(os.path.join(cfg.logdir, subdir), exist_ok=True)
    logger.add(os.path.join(cfg.logdir, "train.log"), level="INFO")
    logger.info(f"Logging to {cfg.logdir}")
    logger.info(OmegaConf.to_yaml(cfg))
    with open(os.path.join(cfg.logdir, "config_train.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


def build_cfg(args):
    cfg_file = OmegaConf.load(args.cfg_file)
    cfg_items, _ = get_cfg_items(cfg_file)
    overrides = [
        f"dataset.seq={args.seq}",
        "train.anim_interval=-1",
    ]

    if args.quick:
        overrides.extend(
            [
                f"train.num_steps={args.quick_steps}",
                f"train.val_interval={max(1, args.quick_steps // 2)}",
                f"train.save_ckpt_interval={args.quick_steps}",
                "train.save_progress_images=true",
                f"train.progress_save_interval={max(1, args.quick_steps // 2)}",
            ]
        )

    overrides.extend(args.override)
    cfg = OmegaConf.merge(default_cfg, cfg_items[0], OmegaConf.from_dotlist(overrides))
    cfg.cfg_file = args.cfg_file
    return cfg


def main(args):
    os.environ.setdefault("TORCH_HOME", "/hdd/u202420081000003/torch_cache")
    os.environ.setdefault("TMPDIR", "/hdd/u202420081000003/tmp")

    # AMASS is only needed for novel-pose animation. The NeuMan training and
    # validation path can run without it.
    gst.get_anim_dataset = lambda cfg: None

    tag = args.exp_name
    if args.quick:
        tag = tag or f"smoke_{args.quick_steps}_noamass_noinit"
        gst.optimize_init = lambda human_gs, num_steps=7000: human_gs
    else:
        tag = tag or "full_noamass"

    cfg = build_cfg(args)
    cfg.exp_name = tag
    safe_state(seed=cfg.seed)
    make_logdir(cfg, tag)

    trainer = gst.GaussianTrainer(cfg)
    trainer.train()
    trainer.save_ckpt()
    trainer.validate()
    trainer.render_full_sequence()

    with open(os.path.join(cfg.logdir, "results_train.json"), "w") as f:
        json.dump(trainer.eval_metrics, f, indent=4)
    print(f"LOGDIR={cfg.logdir}", flush=True)
    print(json.dumps(trainer.eval_metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="lab")
    parser.add_argument("--cfg-file", default="cfg_files/release/neuman/hugs_human_scene.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-steps", type=int, default=100)
    parser.add_argument("--exp-name", default="")
    parser.add_argument("override", nargs="*", help="OmegaConf dotlist overrides, e.g. train.num_steps=1000")
    main(parser.parse_args())
