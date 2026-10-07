"""MC30 evaluation of one accepted checkpoint on one full cohort with the fixed generic sentence."""
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_detail_formal import TrainingPaused, set_random_seed, forward_batch
from research.data import DualScaleDataset, DualScaleTransform
from research_formal.metrics import case_metrics, summarize, write_cases
from .common import GENERIC, SMALL, checkpoint_path, split_run, sha256, write_json, now, read


class GenericPrompt:
    """Every image receives the same class sentence; dataset text is never used."""
    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        sample = dict(self.inner[index])
        sample['text_prompt'] = GENERIC
        return sample


def load_checkpoint(path):
    """The checkpoints were written on Linux; tolerate pickled POSIX path objects on Windows."""
    import os
    import pathlib
    if os.name != 'nt':
        return torch.load(path, map_location='cpu', weights_only=False)
    original = pathlib.PosixPath
    pathlib.PosixPath = pathlib.PurePosixPath
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    finally:
        pathlib.PosixPath = original


def load_model(run, expected_sha256):
    from research_detail_formal.common import load_suite, make_config
    from research_detail_formal.model import build_model
    mode, seed = split_run(run)
    cfg = make_config(load_suite(), run)
    path = checkpoint_path(run)
    digest = sha256(path)
    if digest != expected_sha256:
        raise ValueError(f'{run}: checkpoint hash differs from the accepted audit')
    saved = load_checkpoint(path)
    if saved['epoch'] != cfg.TRAIN.NUM_EPOCHS-1 or saved['progress']['phase'] != 'epoch_complete':
        raise ValueError(f'{run}: not the completed epoch-100 checkpoint')
    stored = saved['research_signature']['config']
    if (stored['RESEARCH']['RUN_ID'], stored['RESEARCH']['QUERY_MODE'], stored['seed'],
            stored['RESEARCH']['GENERIC_PROMPT']) != (run, mode, seed, GENERIC):
        raise ValueError(f'{run}: checkpoint belongs to another run or prompt condition')
    set_random_seed(seed)
    model = build_model(cfg)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.eval().to(cfg.MODEL.DEVICE)
    return model, cfg, digest


def evaluate_cohort(model, cfg, cohort, folder, records, destination, signature, control, groups=None, save_maps=True):
    """Returns the summary. A paused evaluation is rerun from its first batch with the same seed."""
    destination, folder = Path(destination), Path(folder)
    manifest_path = destination/'manifest.json'
    if manifest_path.exists():
        previous = read(manifest_path)
        if previous['signature'] != signature:
            if cohort != 'Val_check' and previous.get('complete'):
                raise ValueError(f'Locked evaluation inputs changed after a completed Test evaluation: {destination}')
        elif previous.get('complete'):
            if sha256(destination/'cases.csv') != previous['cases_sha256']:
                raise ValueError(f'Saved cases were modified: {destination}')
            return read(destination/'summary.json')
    if len({r['Ground Truth'] for r in records}) != len(records) or len({r['Image'] for r in records}) != len(records):
        raise ValueError('Duplicate rows in evaluation list')
    destination.mkdir(parents=True, exist_ok=True)
    manifest = dict(signature=signature, complete=False, started_at=now())
    write_json(manifest_path, manifest)
    if save_maps:
        for sub in ('masks', 'probabilities'):
            (destination/sub).mkdir(exist_ok=True)
    data = GenericPrompt(DualScaleDataset(str(folder), cohort, records, DualScaleTransform(False),
                                          image_size=224, detail_size=448, upsampled=False))
    loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=0)
    set_random_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True  # same inference setting as the accepted Val evaluations
    torch.cuda.reset_peak_memory_stats()
    rows, elapsed = [], 0.
    control.update('running', cohort=cohort, step=0, total_steps=len(loader), cases_completed=0, cases_total=len(records))
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if control.pause_requested():
                control.update('paused', cohort=cohort)
                raise TrainingPaused()
            if any(text != GENERIC for text in batch['text_prompt']):
                raise ValueError('Non-generic prompt reached the model')
            torch.cuda.synchronize()
            start = time.perf_counter()
            draws = torch.sigmoid(forward_batch(model, batch, cfg.MODEL.DEVICE, 30))
            if draws.shape[0] != 30:
                raise ValueError('Expected 30 MC samples')
            mean, variance = draws.mean(0), draws.var(0, unbiased=False)
            torch.cuda.synchronize()
            elapsed += time.perf_counter()-start
            mean, variance = mean.cpu().numpy(), variance.cpu().numpy()
            for b, name in enumerate(batch['mask_name']):
                if not np.isfinite(mean[b]).all():
                    raise ValueError('Non-finite prediction')
                row = dict(name=name, image_name=batch['image_name'][b])
                if save_maps:
                    key = f'{len(rows):06d}'
                    row.update(mask_file=f'masks/{key}.png', probability_file=f'probabilities/{key}.npz')
                    if not cv2.imwrite(str(destination/row['mask_file']), (mean[b] > .5).astype(np.uint8)*255):
                        raise OSError('PNG save failed')
                    np.savez_compressed(destination/row['probability_file'], mean=mean[b].astype(np.float16),
                                        variance=variance[b].astype(np.float16))
                row.update(input_prompt=batch['text_prompt'][b], **case_metrics(folder/'label'/name, mean[b]))
                rows.append(row)
            control.update('running', cohort=cohort, step=index+1, total_steps=len(loader), cases_completed=len(rows),
                           cases_total=len(records))
            del draws
    if len(rows) != len(records) or {r['name'] for r in rows} != {r['Ground Truth'] for r in records}:
        raise RuntimeError('Evaluation cohort coverage mismatch')
    summary = summarize(rows, SMALL)
    if groups:
        by_image = {r['image_name']: r for r in rows}
        summary['unique_image_sensitivity'] = dict(
            n_unique_images=len(groups),
            DSC_percent=100*float(np.mean([np.mean([by_image[i]['DSC'] for i in g]) for g in groups])),
            NSD_percent=100*float(np.mean([np.mean([by_image[i]['NSD'] for i in g]) for g in groups])),
            convention='equal weight per image file SHA256')
    summary.update(run_id=cfg.RESEARCH.RUN_ID, query_mode=cfg.RESEARCH.QUERY_MODE, seed=cfg.seed, cohort=cohort,
                   prompt=GENERIC, mc_samples=30, batch_size=32, inference_seconds=elapsed,
                   milliseconds_per_image=1000*elapsed/len(rows), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   device=torch.cuda.get_device_name(), completed_at=now())
    write_cases(destination/'cases.csv', rows)
    write_json(destination/'summary.json', summary)
    manifest.update(complete=True, n=len(rows), cases_sha256=sha256(destination/'cases.csv'), finished_at=now())
    write_json(manifest_path, manifest)
    return summary
