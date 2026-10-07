"""Evaluate saved research checkpoints; never select a model on target tests."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from train import TrainingControl, TrainingPaused, set_random_seed, forward_batch
from utils.main_utils import CfgNode, read_text
from .common import ROOT, now, write_json, sha256, plain_config, make_signature, assert_signature, object_hash
from .data import make_dataset
from .model import build_model
from .metrics import foreground_threshold, case_metrics, summarize, write_cases


def evaluate(cfg, checkpoint_path, destination, control_directory, target=None, split='Val', beta_override=None,
             diagnostic=False, max_cases=None):
    destination = Path(destination)
    control = TrainingControl(control_directory, variant=cfg.MODEL.VARIANT,
                              dataset=target or cfg.DATASET.NAME, phase='evaluation', step=0)
    if control.pause_requested():
        control.update('paused', exit_code=75)
        raise TrainingPaused()
    control.update('starting', checkpoint=str(checkpoint_path))
    current_signature = make_signature(cfg)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not diagnostic:
        assert_signature(saved.get('research_signature'), current_signature)
        if saved['epoch'] != cfg.TRAIN.NUM_EPOCHS-1 or saved['progress']['phase'] != 'epoch_complete':
            raise ValueError('Formal evaluation requires the completed final epoch checkpoint')
    threshold = foreground_threshold(cfg)
    test_cfg = CfgNode(plain_config(cfg))
    if target and target != cfg.DATASET.NAME:
        test_cfg.DATASET.NAME = target
        for key, directory in [('TRAIN_PATH', 'Train_Folder'), ('VAL_PATH', 'Val_Folder'), ('TEST_PATH', 'Test_Folder'), ('TEXT_PROMPT_PATH', 'Prompts_Folder')]:
            test_cfg.DATASET[key] = str(ROOT / 'data' / target / directory) + '/'
    text_name = 'Test_text_original.xlsx' if split == 'Test' else 'Val_text.xlsx'
    text_path = Path(test_cfg.DATASET.TEXT_PROMPT_PATH) / text_name
    records = read_text(text_path)
    if len({r['Ground Truth'] for r in records}) != len(records):
        raise ValueError('Duplicate mask in evaluation cohort')
    if max_cases is not None:
        records = records[:max_cases]
    data = make_dataset(test_cfg, split, records)
    folder = Path(getattr(test_cfg.DATASET, split.upper() + '_PATH'))
    fingerprints = {}
    for row in records:
        for kind, key in [('img', 'Image'), ('label', 'Ground Truth')]:
            path = folder / kind / row[key]
            fingerprints[str(path)] = sha256(path)
    evaluation_signature = dict(training_signature=current_signature['digest'], checkpoint_sha256=sha256(checkpoint_path),
                                prompt_sha256=sha256(text_path), evaluation_files=fingerprints,
                                target=test_cfg.DATASET.NAME, split=split, seed=cfg.seed,
                                mc_samples=cfg.TEST.NUM_SAMPLES, batch_size=32, beta_override=beta_override,
                                diagnostic=diagnostic, max_cases=max_cases)
    digest = object_hash(evaluation_signature)
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['digest'] != digest:
            raise ValueError(f'Evaluation inputs changed; choose a fresh output directory: {destination}')
        if previous.get('complete'):
            summary = json.loads((destination / 'summary.json').read_text())
            if sha256(destination / 'cases.csv') != previous['cases_sha256']:
                raise ValueError('Saved evaluation CSV was modified')
            return summary
    manifest = dict(digest=digest, signature=evaluation_signature, complete=False, started_at=now())
    destination.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    for subdir in ('masks', 'probabilities'):
        (destination / subdir).mkdir(exist_ok=True)
    set_random_seed(cfg.seed)
    model = build_model(cfg)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    if beta_override is not None:
        # Diagnose only the visual-key direction targeted by the proposed method.
        for adapter in model.pvl_adapters:
            adapter.two_way.cross_attn_txt_to_img.beta = beta_override
    model.eval().to(cfg.MODEL.DEVICE)
    set_random_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True  # preserve the author's inference setting
    loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=0)
    rows, inference_seconds = [], 0.
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if control.pause_requested():
                control.update('paused', exit_code=75, reason='Resume reruns this evaluation with the original RNG seed.')
                raise TrainingPaused()
            torch.cuda.synchronize()
            start = time.perf_counter()
            draws = torch.sigmoid(forward_batch(model, batch, cfg.MODEL.DEVICE, cfg.TEST.NUM_SAMPLES))
            mean = draws.mean(0)
            variance = draws.var(0, unbiased=False)
            entropy = -(mean*torch.log(mean+1e-8)+(1-mean)*torch.log(1-mean+1e-8))
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter()-start
            mean, variance, entropy = (a.cpu().numpy() for a in (mean, variance, entropy))
            beta = getattr(model, 'last_beta', None)
            beta = beta.cpu().numpy() if beta is not None else None
            for b, name in enumerate(batch['mask_name']):
                key = f'{len(rows):06d}'
                mask_file, probability_file = f'masks/{key}.png', f'probabilities/{key}.npz'
                if not cv2.imwrite(str(destination / mask_file), (mean[b] > .5).astype(np.uint8)*255):
                    raise OSError('Failed to save lossless prediction')
                values = dict(mean=mean[b], variance=variance[b], entropy=entropy[b])
                if beta is not None:
                    values['beta_mean_by_layer_patch'] = beta[b]
                np.savez_compressed(destination / probability_file, **values)
                row = dict(name=name, image_name=batch['image_name'][b], mask_file=mask_file,
                           probability_file=probability_file, **case_metrics(folder / 'label' / name, mean[b]))
                rows.append(row)
            control.update('running', phase='evaluation', step=index+1, total_steps=len(loader),
                           cases_completed=len(rows), inference_seconds=inference_seconds)
            del draws
    if len(rows) != len(records) or {r['name'] for r in rows} != {r['Ground Truth'] for r in records}:
        raise RuntimeError('Evaluation cohort coverage mismatch')
    summary = summarize(rows, threshold)
    summary.update(variant=cfg.MODEL.VARIANT, source=cfg.DATASET.NAME, target=test_cfg.DATASET.NAME, split=split,
                   mc_samples=cfg.TEST.NUM_SAMPLES, inference_seconds=inference_seconds,
                   milliseconds_per_image=1000*inference_seconds/len(rows),
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   total_parameters=sum(p.numel() for p in model.parameters()),
                   checkpoint_sha256=evaluation_signature['checkpoint_sha256'],
                   diagnostic=diagnostic, subset_check=max_cases is not None, completed_at=now())
    write_cases(destination / 'cases.csv', rows)
    write_json(destination / 'summary.json', summary)
    manifest.update(complete=True, finished_at=now(), n=len(rows), cases_sha256=sha256(destination / 'cases.csv'))
    write_json(manifest_path, manifest)
    control.update('completed', phase='evaluation', cases_completed=len(rows), exit_code=0)
    return summary
