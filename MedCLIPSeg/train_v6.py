from local_runtime import configure_local_models
configure_local_models()

import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.dataloader import DatasetSegmentation, RandomGenerator, ValGenerator
from trainers import *
import os
import argparse
import random
import numpy as np
from torch.nn.modules.loss import BCEWithLogitsLoss
import logging
import json
import sys
from datetime import datetime, timezone
from functools import partial
from utils.main_utils import load_cfg_from_cfg_file, read_text

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--config-file",
    required=True,
    type=str,
    help="Path to config file",
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help="Whether to resume training"
    )
    parser.add_argument('--init-from', type=str, default=None,
                        help='Research only: initialize model weights, with a fresh optimizer; mutually exclusive with --resume.')

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--data_percentage",
        type=int,
        default=100,
        help="Percentage of data to use.")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="output directory")

    parser.add_argument(
        '--num-workers',
        type=int,
        default=0 if os.name == 'nt' else 8,
        help="DataLoader workers (Windows default: 0; other platforms: 8)."
    )

    parser.add_argument(
        '--gradient-checkpointing',
        action='store_true',
        help="Recompute frozen encoder activations to reduce training GPU memory."
    )

    parser.add_argument(
        '--control-dir',
        type=str,
        default=os.path.join('..', 'control'),
        help="Directory containing PAUSE.request and trainer_status.json."
    )

    parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="modify config options using the command-line",
        )

    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error('--resume and --init-from are mutually exclusive')

    cfg = load_cfg_from_cfg_file(args.config_file)

    cfg.merge_from_list(args.opts)

    cfg.update({k: v for k, v in vars(args).items()})
    cfg.MODEL.GRAD_CHECKPOINTING = args.gradient_checkpointing

    return cfg


def print_args(args, cfg):
    logging.info("***************")
    logging.info("** Arguments **")
    logging.info("***************")
    logging.info("************")
    logging.info("** Config **")
    logging.info("************")
    logging.info(cfg)

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def calc_loss(low_res_logits, low_res_label_batch, ce_loss, dice_loss, cfg):
    loss_ce = ce_loss(low_res_logits, low_res_label_batch.float())
    if getattr(cfg.TRAIN, 'DICE_MODE', 'legacy') == 'image':
        low_res_logits = low_res_logits.unsqueeze(1) if low_res_logits.ndim == 3 else low_res_logits
        low_res_label_batch = low_res_label_batch.unsqueeze(1) if low_res_label_batch.ndim == 3 else low_res_label_batch
    loss_dice = dice_loss(low_res_logits, low_res_label_batch)
    loss = cfg.TRAIN.DICE_WEIGHT * loss_dice + cfg.TRAIN.CE_WEIGHT * loss_ce
    return loss


def is_research(cfg):
    return getattr(cfg.MODEL, 'VARIANT', 'official') != 'official'


def forward_batch(model, batch, device, num_samples=None):
    kwargs = {'image': batch['image'].to(device), 'text': batch['text_prompt']}
    if 'image_detail' in batch:
        kwargs['image_detail'] = batch['image_detail'].to(device)
    if num_samples is not None:
        kwargs['num_samples'] = num_samples
    return model(**kwargs)

PAUSED_EXIT_CODE = 75


class TrainingPaused(Exception):
    """A requested pause completed after a durable checkpoint was written."""


class TrainingControl:
    def __init__(self, directory, **initial_status):
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)
        self.state = dict(initial_status)

    def pause_requested(self):
        return os.path.isfile(os.path.join(self.directory, "PAUSE.request"))

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


def worker_init_fn(worker_id, seed):
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def replay_iterator(dataloader, start_rng_state, next_batch_index=0, current_rng_state=None):
    """Recreate shuffle/iterator state without repeating any optimizer update.

    Exact batch replay requires num_workers=0. Data loading and augmentation for
    consumed batches are replayed, then all RNGs return to the saved boundary.
    """
    if current_rng_state is None:
        if next_batch_index:
            raise ValueError("Cannot resume consumed batches without a saved RNG state")
        return iter(dataloader)
    if dataloader.num_workers != 0:
        raise ValueError("Exact mid-epoch resume requires --num-workers 0")
    if next_batch_index == 0:
        # Batch-zero pauses are taken before iter(dataloader) consumes its
        # base seed. Do not undo that seed consumption after constructing it.
        restore_rng_state(current_rng_state)
        return iter(dataloader)
    restore_rng_state(start_rng_state)
    iterator = iter(dataloader)
    for _ in range(next_batch_index):
        try:
            next(iterator)
        except StopIteration as exc:
            raise ValueError("Checkpoint batch position exceeds this dataset") from exc
    restore_rng_state(current_rng_state)
    return iterator


def atomic_save_checkpoint(checkpoint, path):
    temporary = path + f".tmp-{os.getpid()}"
    with open(temporary, "wb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def evaluate_validation_loss(model, val_dataloader, device, ce_loss, dice_loss, cfg,
                             progress, on_batch, current_rng_state=None):
    model.eval()
    val_losses = progress["val_losses"]
    dice_scores = progress["dice_scores"]
    next_batch_index = progress["next_batch_index"]
    iterator = replay_iterator(val_dataloader, progress["validation_start_rng_state"],
                               next_batch_index, current_rng_state)

    with torch.no_grad():
        for index, batch in enumerate(tqdm(iterator, total=len(val_dataloader),
                                          initial=next_batch_index, desc="Validation"),
                                      start=next_batch_index):
            images = batch["image"].to(device)
            masks = batch["ground_truth_mask"].to(device)
            text = batch["text_prompt"]

            logits = forward_batch(model, batch, device, num_samples=1)[0]
            loss = calc_loss(logits, masks, ce_loss, dice_loss, cfg)
            val_losses.append(loss.item())

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            if preds.ndim == 3:
                preds = preds.unsqueeze(1)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_scores.extend(dice.cpu().numpy())
            progress["next_batch_index"] = index + 1
            on_batch(progress)

    model.train()
    return mean(val_losses), mean(dice_scores)


def run_training(cfg, control, checkpoint_dir, results_name):
    import time
    research = is_research(cfg)
    if research:
        from pathlib import Path
        from research_v6.common import ROOT
        if any(Path(checkpoint_dir).resolve().is_relative_to(ROOT / name)
               for name in ('outputs_local_training', 'outputs_medclipseg')):
            raise ValueError('Research checkpoints must not overwrite a reproduction output directory')
    if getattr(cfg, 'init_from', None) and (cfg.resume or not research):
        raise ValueError('--init-from is only valid for a new research run')
    logger = logger_config(os.path.join(checkpoint_dir, "log.txt"))
    logger.info("************\n** Config **\n************")
    logger.info(cfg)
    if cfg.seed >= 0:
        logger.info("Setting fixed seed: {}".format(cfg.seed))
        set_random_seed(cfg.seed)

    ce_loss = BCEWithLogitsLoss()
    dice_mode = getattr(cfg.TRAIN, 'DICE_MODE', 'legacy')
    if dice_mode not in ('legacy', 'image'):
        raise ValueError(f'Unknown Dice mode: {dice_mode}')
    dice_loss = monai.losses.DiceLoss(include_background=dice_mode == 'image', sigmoid=True, reduction="mean")
    train_tf = transforms.Compose([RandomGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])])
    val_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
    train_text_file = f"Train_text_{cfg.data_percentage}.xlsx" if cfg.data_percentage != 100 else "Train_text.xlsx"
    val_text_file = f"Val_text_{cfg.data_percentage}.xlsx" if cfg.data_percentage != 100 else "Val_text.xlsx"
    train_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + train_text_file)
    val_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + val_text_file)
    train_dataset = DatasetSegmentation(cfg.DATASET.TRAIN_PATH, cfg.DATASET.NAME, train_text, train_tf,
                                        image_size=cfg.DATASET.SIZE)
    val_dataset = DatasetSegmentation(cfg.DATASET.VAL_PATH, cfg.DATASET.NAME, val_text, val_tf,
                                      image_size=cfg.DATASET.SIZE)
    if research:
        from research_v6.data import make_dataset
        train_dataset, val_dataset = make_dataset(cfg, 'Train', train_text), make_dataset(cfg, 'Val', val_text)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                 worker_init_fn=partial(worker_init_fn, seed=cfg.seed),
                                 num_workers=cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                               worker_init_fn=partial(worker_init_fn, seed=cfg.seed),
                               num_workers=cfg.num_workers, pin_memory=True)

    signature = None
    if research:
        from research_v6.model import build_model, initialize_from_checkpoint
        from research_v6.common import make_signature, assert_signature, optimizer_groups, write_json
        signature = make_signature(cfg)
        model = build_model(cfg)
        if not cfg.resume:
            missing = initialize_from_checkpoint(model, getattr(cfg, 'init_from', None) or cfg.RESEARCH.INIT_FROM)
            logger.info(f'Initialized shared weights; fresh parameters: {missing}')
        # New-module initialization must not change loader shuffle or MC seed setup.
        set_random_seed(cfg.seed)
    elif cfg.MODEL.CLIP_MODEL == "unimedclip":
        model = build_medclipseg_unimedclip(cfg)
    elif cfg.MODEL.CLIP_MODEL == "biomedclip":
        model = build_medclipseg_biomedclip(cfg)
    elif cfg.MODEL.CLIP_MODEL == "clip":
        model = build_medclipseg_clip(cfg)
    elif cfg.MODEL.CLIP_MODEL == "pubmedclip":
        model = build_medclipseg_pubmedclip(cfg)
    else:
        raise ValueError(f"Unsupported CLIP model: {cfg.MODEL.CLIP_MODEL}")

    enabled = {name for name, param in model.named_parameters() if param.requires_grad}
    logger.info(f"Parameters to be updated: {enabled}")
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.to(cfg.MODEL.DEVICE)
    parameters = optimizer_groups(model, cfg) if research else filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=cfg.TRAIN.LEARNING_RATE)
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,
                                                          eta_min=getattr(cfg.TRAIN, 'MIN_LEARNING_RATE', 1e-4))
    resume_path = os.path.join(checkpoint_dir, f"{results_name}_latest.pth")
    best_path = os.path.join(checkpoint_dir, f"{results_name}_best_dice.pth")
    runtime = {"num_workers": cfg.num_workers, "batch_size": cfg.TRAIN.BATCH_SIZE,
               "train_size": len(train_dataset), "val_size": len(val_dataset),
               "gradient_checkpointing": bool(getattr(cfg.MODEL, "GRAD_CHECKPOINTING", False))}
    start_epoch = 0
    best_loss = float("inf")
    best_dice = 0
    resume_progress = None
    resume_rng_state = None

    if cfg.resume:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Requested resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if research:
            assert_signature(checkpoint.get('research_signature'), signature)
            if checkpoint.get('runtime') != runtime:
                raise ValueError('Research runtime changed; exact continuation is not allowed')
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            logger.warning("Legacy checkpoint lacks scheduler state; continuation cannot be exact.")
        best_loss = checkpoint.get("best_loss", float("inf"))
        best_dice = checkpoint.get("best_dice", 0.0)
        if "best_dice" not in checkpoint:
            logger.warning("Legacy checkpoint lacks best_dice; best-model tracking restarts at 0.")
        resume_progress = checkpoint.get("progress")
        partial_epoch = resume_progress and resume_progress["phase"] != "epoch_complete"
        start_epoch = checkpoint["epoch"] if partial_epoch else checkpoint["epoch"] + 1
        if partial_epoch:
            saved_runtime = dict(checkpoint.get("runtime", {}))
            if "gradient_checkpointing" not in saved_runtime:
                logger.warning("Legacy runtime metadata lacks gradient_checkpointing; assuming False.")
                saved_runtime["gradient_checkpointing"] = False
            if cfg.num_workers != 0 or saved_runtime != runtime:
                raise ValueError("Mid-epoch resume needs --num-workers 0 and unchanged batch size, "
                                 "dataset lengths and gradient-checkpointing setting")
            if "rng_state" not in checkpoint:
                raise ValueError("Mid-epoch checkpoint is missing RNG state")
            resume_rng_state = checkpoint["rng_state"]
        else:
            resume_progress = None
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
        else:
            logger.warning("Legacy checkpoint lacks RNG state; continuation is not an exact restart.")
        logger.info(f"Loaded checkpoint: next epoch={start_epoch + 1}, partial_epoch={bool(partial_epoch)}")
        del checkpoint

    model.train()
    epoch_durations = []
    epoch_clock = time.perf_counter()
    if research:
        write_json(os.path.join(checkpoint_dir, 'research_manifest.json'), signature)

    def make_checkpoint(epoch, progress):
        saved = {"checkpoint_version": 2, "model": model.state_dict(), "epoch": epoch,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_loss": best_loss, "best_dice": best_dice,
                "rng_state": capture_rng_state(), "progress": progress, "runtime": runtime}
        if research:
            saved.update(checkpoint_version=3, research_signature=signature)
        return saved

    def report_batch(epoch, progress):
        phase = progress["phase"]
        step = progress["next_batch_index"]
        extra = {}
        if research:
            extra = dict(variant=cfg.MODEL.VARIANT, total_epochs=num_epochs,
                         latest_loss=progress['epoch_losses'][-1] if progress['epoch_losses'] else None,
                         gpu_allocated_mib=torch.cuda.memory_allocated()/2**20 if torch.cuda.is_available() else 0,
                         remaining_seconds=mean(epoch_durations[-5:])*(num_epochs-epoch) if epoch_durations else None)
            if getattr(model, 'last_diagnostics', None):
                extra['branch_diagnostics'] = model.last_diagnostics
        control.update("running", phase=phase, epoch=epoch + 1, step=step, **extra,
                       total_steps=len(train_dataloader) if phase == "train" else len(val_dataloader))
        if control.pause_requested():
            if cfg.num_workers != 0:
                control.update("pause_pending", reason="Workers > 0: pause deferred to the epoch boundary")
                return
            control.update("pausing", reason="Saving a checkpoint at the completed batch boundary")
            atomic_save_checkpoint(make_checkpoint(epoch, progress), resume_path)
            control.update("paused", checkpoint=os.path.abspath(resume_path), exit_code=PAUSED_EXIT_CODE)
            logger.info(f"Paused after epoch {epoch + 1}, {phase} batch {step}: {resume_path}")
            raise TrainingPaused()

    for epoch in range(start_epoch, num_epochs):
        epoch_clock = time.perf_counter()
        resumed_partial_epoch = resume_progress is not None
        progress = resume_progress if resume_progress is not None else {
            "phase": "train", "next_batch_index": 0, "epoch_losses": [],
            "epoch_start_rng_state": capture_rng_state(),
        }
        resume_progress = None
        if progress["phase"] == "train":
            # Before the first batch this can also pause after a slow model load.
            report_batch(epoch, progress)
            next_batch_index = progress["next_batch_index"]
            iterator = replay_iterator(train_dataloader, progress["epoch_start_rng_state"],
                                       next_batch_index, resume_rng_state)
            resume_rng_state = None
            for index, batch in enumerate(tqdm(iterator, total=len(train_dataloader), initial=next_batch_index),
                                          start=next_batch_index):
                from research_v6.objective import paired_objective
                optimizer.zero_grad()
                loss, diagnostics = paired_objective(model, batch, cfg, ce_loss, dice_loss, forward_batch, calc_loss)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite paired loss')
                loss.backward()
                optimizer.step()
                model.last_diagnostics = diagnostics
                progress["epoch_losses"].append(loss.item())
                progress["next_batch_index"] = index + 1
                report_batch(epoch, progress)
            # Exactly one scheduler step, including resumes after the final train batch.
            scheduler.step()
            progress.update(phase="validation", next_batch_index=0, val_losses=[], dice_scores=[],
                            validation_start_rng_state=capture_rng_state())

        report_batch(epoch, progress)
        mean_val_loss, mean_val_dice = evaluate_validation_loss(
            model, val_dataloader, cfg.MODEL.DEVICE, ce_loss, dice_loss, cfg,
            progress, lambda state: report_batch(epoch, state), resume_rng_state)
        resume_rng_state = None
        mean_epoch_loss = mean(progress["epoch_losses"])
        logger.info(f'EPOCH: {epoch+1} | Training Loss: {mean_epoch_loss:.4f} | Validation Loss: {mean_val_loss:.4f}')
        if research and getattr(model, 'last_diagnostics', None):
            logger.info('LAST_TRAIN_BATCH_DIAGNOSTICS: '+json.dumps(model.last_diagnostics))
        best_loss = min(best_loss, mean_val_loss)
        improved = mean_val_dice > best_dice
        if improved:
            logger.info(f"New best Dice: {best_dice:.4f} -> {mean_val_dice:.4f}")
            best_dice = mean_val_dice
        else:
            logger.info(f"Dice: {mean_val_dice:.4f}")
        progress["phase"] = "epoch_complete"
        checkpoint = make_checkpoint(epoch, progress)
        if improved:
            atomic_save_checkpoint(checkpoint, best_path)
        atomic_save_checkpoint(checkpoint, resume_path)
        del checkpoint
        if research and not resumed_partial_epoch:
            epoch_durations.append(time.perf_counter()-epoch_clock)
        control.update("running", phase="epoch_complete", epoch=epoch + 1,
                       checkpoint=os.path.abspath(resume_path), validation_loss=float(mean_val_loss),
                       validation_dice=float(mean_val_dice))
        if control.pause_requested():
            control.update("paused", exit_code=PAUSED_EXIT_CODE)
            raise TrainingPaused()
    control.update("completed", phase="completed", epoch=num_epochs, checkpoint=os.path.abspath(resume_path), exit_code=0)
    return 0


def main():
    cfg = get_arguments()
    if cfg.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    cfg.DATASET.NAME = cfg.DATASET.NAME + f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else cfg.DATASET.NAME
    checkpoint_dir = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    results_name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{cfg.MODEL.BACKBONE.replace('/', '-')}"
    checkpoint_path = os.path.abspath(os.path.join(checkpoint_dir, f"{results_name}_latest.pth"))
    control = TrainingControl(cfg.control_dir, dataset=cfg.DATASET.NAME, seed=cfg.seed,
                              checkpoint=checkpoint_path, epoch=None, step=None)
    if control.pause_requested():
        control.update("paused", reason="PAUSE.request exists before startup", exit_code=PAUSED_EXIT_CODE)
        return PAUSED_EXIT_CODE
    control.update("starting", phase="initialization")
    try:
        return run_training(cfg, control, checkpoint_dir, results_name)
    except TrainingPaused:
        return PAUSED_EXIT_CODE
    except Exception as exc:
        control.update("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
