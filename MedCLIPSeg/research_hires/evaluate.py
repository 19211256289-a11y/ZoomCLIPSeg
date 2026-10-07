"""Final-epoch source-Val evaluation (MC30, fixed sentence). Predictions are thresholded at the model's output size."""
import torch

from train_detail_formal import set_random_seed
from research_detail_ext.evaluate import evaluate_cohort   # generic over model, prompt and threshold; used read-only
from .common import checkpoint_path, make_config, split_run, sha256
from .model import build_for


def inspect_checkpoint(suite, run):
    arm, seed = split_run(suite, run)
    cfg = make_config(suite, run)
    path = checkpoint_path(suite, run)
    digest = sha256(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['epoch'] != cfg.TRAIN.NUM_EPOCHS-1 or saved['progress']['phase'] != 'epoch_complete':
        raise ValueError(f'{run}: not the completed final-epoch checkpoint')
    stored = saved['research_signature']['config']
    if (stored['RESEARCH']['RUN_ID'], stored['RESEARCH']['HIRES_MODE'], stored['seed'], stored['RESEARCH']['SUITE']) != \
            (run, suite['arms'][arm]['hires'], seed, suite['name']):
        raise ValueError(f'{run}: checkpoint belongs to another run')
    return saved, cfg, digest


def load_model(suite, run, expected_sha256):
    saved, cfg, digest = inspect_checkpoint(suite, run)
    if digest != expected_sha256:
        raise ValueError(f'{run}: final checkpoint changed after training completed')
    set_random_seed(cfg.seed)
    model = build_for(cfg)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.eval().to(cfg.MODEL.DEVICE)
    return model, cfg, digest
