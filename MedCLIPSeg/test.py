from local_runtime import configure_local_models
configure_local_models()

import os
import cv2
import torch
import argparse
import logging
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.dataloader import DatasetSegmentation, ValGenerator
from trainers import *
from utils.main_utils import load_cfg_from_cfg_file, read_text, normalize
import matplotlib.pyplot as plt
import json
import sys
from datetime import datetime, timezone

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--config-file",
    required=True,
    type=str,
    help="Path to config file",
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        '--prompt_design',
        type=str,
        default="original",
        help="Text prompt design."
    )

    parser.add_argument(
        "--data_percentage",
        type=int,
        default=100,
        help="Percentage of data to use.")

    parser.add_argument(
        "--source_dataset",
        type=str,
        help="source dataset name for loading trained model.")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="output directory")

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help="Inference batch size (official default: 32)."
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=0 if os.name == 'nt' else 8,
        help="DataLoader workers (Windows default: 0; other platforms: 8)."
    )

    parser.add_argument(
        '--control-dir',
        type=str,
        default=None,
        help="Optional training-pipeline control directory for safe inference pauses."
    )

    parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="modify config options using the command-line",
        )

    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.config_file)

    cfg.merge_from_list(args.opts)

    cfg.update({k: v for k, v in vars(args).items()})

    return cfg

def logger_config(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

class EvaluationControl:
    def __init__(self, directory, cfg):
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)
        self.state = {"phase": "evaluation", "dataset": cfg.DATASET.NAME,
                      "percentage": cfg.data_percentage, "seed": cfg.seed,
                      "epoch": None, "step": 0, "checkpoint": None}

    def update(self, status, **fields):
        self.state.update(fields)
        self.state.update(status=status, pid=os.getpid(),
                          updated_at=datetime.now(timezone.utc).isoformat())
        target = os.path.join(self.directory, "trainer_status.json")
        temporary = target + f".tmp-{os.getpid()}"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.state, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)

    def pause_requested(self):
        if not os.path.isfile(os.path.join(self.directory, "PAUSE.request")):
            return False
        self.update("paused", exit_code=75,
                    reason="训练 checkpoint 已保存，继续时本次推理从头重跑以保持 seed/MC 序列。")
        return True


def run_evaluation(cfg, control=None):

    if cfg.seed >= 0:
        print(f"Setting fixed seed: {cfg.seed}")
        set_random_seed(cfg.seed)

    cfg.DATASET.NAME = cfg.DATASET.NAME+f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else cfg.DATASET.NAME
    results_root = os.path.join(cfg.output_dir, cfg.DATASET.NAME, f"seg_results", f"seed{cfg.seed}")
    os.makedirs(results_root, exist_ok=True)
    logger = logger_config(os.path.join(results_root, "log.txt"))

    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)

    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
    results_name = (
    f"MedCLIPSeg_"
    f"{cfg.MODEL.CLIP_MODEL}_"
    f"{backbone_name}"
    )

    checkpoint_type = "latest" if cfg.TEST.USE_LATEST else "best_dice"
    checkpoint_path = os.path.join(
        cfg.output_dir,
        (cfg.source_dataset or cfg.DATASET.NAME) if cfg.data_percentage == 100 else cfg.DATASET.NAME,
        "trained_models",
        f"seed{cfg.seed}",
        f"{results_name}_{checkpoint_type}.pth"
    )
    if control:
        control.update("starting", checkpoint=os.path.abspath(checkpoint_path))
        if control.pause_requested():
            return 75

    if(cfg.MODEL.CLIP_MODEL == "unimedclip"):
        model = build_medclipseg_unimedclip(cfg)
    elif(cfg.MODEL.CLIP_MODEL == "biomedclip"):
        model = build_medclipseg_biomedclip(cfg)
    elif(cfg.MODEL.CLIP_MODEL == "clip"):
        model = build_medclipseg_clip(cfg)
    elif(cfg.MODEL.CLIP_MODEL == "pubmedclip"):
        model = build_medclipseg_pubmedclip(cfg)

    checkpoint = torch.load(checkpoint_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(cfg.MODEL.DEVICE)

    test_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
    test_text_file = f"Test_text_{cfg.prompt_design}.xlsx"
    test_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + test_text_file)

    test_dataset = DatasetSegmentation(cfg.DATASET.TEST_PATH, cfg.DATASET.NAME,
                                        test_text, test_tf, image_size=cfg.DATASET.SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers)
    if control:
        control.update("running", total_steps=len(test_dataloader))
        if control.pause_requested():
            return 75

    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader), start=1):
            if control and control.pause_requested():
                return 75

            seg_samples = model(image=batch["image"].to(cfg.MODEL.DEVICE),
                                text=batch["text_prompt"], num_samples=cfg.TEST.NUM_SAMPLES)  # (B, num_classes, H, W)

            seg_samples = torch.sigmoid(seg_samples)

            seg_logits = seg_samples.mean(dim=0)             # predictive mean
            seg_unc = - (seg_logits * torch.log(seg_logits + 1e-8) +
                 (1 - seg_logits) * torch.log(1 - seg_logits + 1e-8))

            mask_preds = (seg_logits > 0.5)

            dataset_names = batch["dataset_name"]   # list of strings
            mask_names = batch["mask_name"]         # list of strings

            for i in range(len(dataset_names)):

                pred_mask = mask_preds[i].cpu().numpy().astype(np.uint8)

                dataset_name = dataset_names[i]
                mask_name = mask_names[i]

                # Then in the loop:
                binary_pred = np.uint8(pred_mask > 0)

                save_dir = os.path.join(cfg.output_dir,
                                        cfg.DATASET.NAME,
                                        f"seg_results",
                                        f"seed{cfg.seed}",
                                        results_name + f"_Prompt-{cfg.prompt_design}")

                save_unc_dir = os.path.join(cfg.output_dir,
                                        cfg.DATASET.NAME,
                                        f"unc_results",
                                        f"seed{cfg.seed}",
                                        results_name + f"_Prompt-{cfg.prompt_design}")

                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(save_unc_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, mask_name), binary_pred * 255)

                u_map = seg_unc[i].cpu().numpy()
                u_map = normalize(u_map)
                colormap = plt.get_cmap('nipy_spectral')
                u_map_color = (colormap(u_map)[:, :, :3] * 255).astype(np.uint8)
                u_map_colored = cv2.cvtColor(u_map_color, cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join(save_unc_dir, mask_name), u_map_colored)
            if control:
                control.update("running", step=step)
                if control.pause_requested():
                    return 75

    if control:
        control.update("completed", exit_code=0)
    return 0


def main():
    cfg = get_arguments()
    control = EvaluationControl(cfg.control_dir, cfg) if cfg.control_dir else None
    if control:
        control.update("starting")
        if control.pause_requested():
            return 75
    try:
        return run_evaluation(cfg, control)
    except Exception as exc:
        if control:
            control.update("failed", error=f"{type(exc).__name__}: {exc}")
        raise

if __name__ == "__main__":
    sys.exit(main())
