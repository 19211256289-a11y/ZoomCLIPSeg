"""One checkpoint on one cohort: single pass and reflection averaging with 5 to 30 stochastic passes, and the locked
two-pass rule on top of either. The first pass with 30 passes is the accepted evaluation (same data, seeding, batches)."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import research_zoom.evaluate as accepted
from research_zoom.geometry import region, window_rule, crop, compose
from research_formal.metrics import case_metrics
from research_zoom_probe.evaluate import save_rows, done, finish
from train_detail_formal import TrainingPaused, set_random_seed, forward_batch
from .common import write_json, now, PASSES, SINGLE, HFLIP, SCZ, SCZFLIP, CONFIGS, passes_of


def draws_of(model, cfg, batch, passes):
    """[passes, B, H, W] probability maps of independent stochastic forward passes (the accepted call for 30)."""
    draws = torch.sigmoid(forward_batch(model, batch, cfg.MODEL.DEVICE, passes))
    if draws.dim() != 4 or draws.shape[0] != passes:
        raise ValueError(f'Unexpected shape of the stochastic passes: {tuple(draws.shape)}')
    if not torch.isfinite(draws).all():
        raise ValueError('Non-finite prediction')
    return draws


def mirrored(batch):
    return dict(batch, image=batch['image'].flip(-1), image_detail=batch['image_detail'].flip(-1))


def as_numpy(maps):
    return maps.float().cpu().numpy().astype(np.float32)


def budget(model, cfg, folder, records, prompt, protocol, destination, signature, control, label):
    """-> cases.csv with one row per image and configuration (see common.CONFIGS)."""
    folder, destination = Path(folder), Path(destination)
    if done(destination, signature, 'cases.csv'):
        return
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'manifest.json', dict(signature=signature, complete=False, started_at=now()))
    batch_size, tau, zoom = protocol['batch_size'], protocol['target_fraction'], protocol['zoom']
    state = dict(stage='first pass', step=0)

    def tick():
        if control.pause_requested():
            control.update('paused', cohort=label)
            raise TrainingPaused()
        state['step'] += 1
        control.update('running', cohort=label, config=state['stage'], step=state['step'], total_steps=None)

    loader = DataLoader(accepted.dataset(folder, records), batch_size=batch_size, shuffle=False, num_workers=0)
    torch.backends.cudnn.benchmark = True
    image_of, plain, flip = {}, {k: {} for k in PASSES}, {k: {} for k in PASSES}
    for stage, store, transform in (('first pass', plain, None), ('first pass, reflected', flip, mirrored)):
        state['stage'] = stage
        if transform is None:
            set_random_seed(cfg.seed)       # the reflected pass continues the random stream, so that its noise is independent
        with torch.no_grad():
            for batch in loader:
                tick()
                if any(text != prompt for text in batch['text_prompt']):
                    raise ValueError('Unexpected prompt reached the model')
                draws = draws_of(model, cfg, transform(batch) if transform else batch, 30)
                if transform:
                    draws = draws.flip(-1)
                for k in PASSES:
                    mean = as_numpy(draws[:k].mean(0))
                    for b, name in enumerate(batch['mask_name']):
                        store[k][name] = mean[b]
                for b, name in enumerate(batch['mask_name']):
                    image_of[name] = batch['image_name'][b]
    names = list(image_of)
    if len(names) != len(records) or any(len(plain[k]) != len(names) or len(flip[k]) != len(names) for k in PASSES):
        raise RuntimeError('Cohort coverage mismatch')

    base, measured, rows = {}, {}, []
    for config in SINGLE+HFLIP:
        k = passes_of(config)
        base[config] = plain[k] if config in SINGLE else {n: (plain[k][n]+flip[k][n])/2 for n in names}
        measured[config] = {n: case_metrics(folder/'label'/n, np.ascontiguousarray(base[config][n], dtype=np.float32)) for n in names}
        rows += [dict(config=config, name=n, image_name=image_of[n], kind='none', window=1., fallback=0, **measured[config][n])
                 for n in names]

    for config in SCZ+SCZFLIP:
        k, both = passes_of(config), config in SCZFLIP
        start_from = f'hflip{k:02d}' if both else f'single{k:02d}'
        state['stage'] = config
        todo = []
        for n in names:
            fraction, box = region(base[start_from][n] > .5)
            kind, side = window_rule(fraction, box, tau, zoom, False)
            if kind == 'none':
                rows.append(dict(config=config, name=n, image_name=image_of[n], kind='none', window=1., fallback=0,
                                 **measured[start_from][n]))
            else:
                todo.append((n, kind, side, box))
        set_random_seed(cfg.seed)
        with torch.no_grad():
            for start in range(0, len(todo), batch_size):
                tick()
                chunk = todo[start:start+batch_size]
                images, details, rects, shapes = [], [], [], []
                for n, kind, side, box in chunk:
                    raw = cv2.imread(str(folder/'img'/image_of[n]))
                    window, rect = crop(raw, kind, side, box)
                    image, detail = accepted.tensors(window)
                    images.append(image)
                    details.append(detail)
                    rects.append(rect)
                    shapes.append(raw.shape[:2])
                batch = dict(image=torch.stack(images), image_detail=torch.stack(details), text_prompt=[prompt]*len(chunk))
                mean = draws_of(model, cfg, batch, k).mean(0)
                if both:
                    mean = (mean+draws_of(model, cfg, mirrored(batch), k).flip(-1).mean(0))/2
                mean = as_numpy(mean)
                for b, (n, kind, side, box) in enumerate(chunk):
                    final = compose(mean[b], kind, rects[b], shapes[b])
                    if final is None:
                        rows.append(dict(config=config, name=n, image_name=image_of[n], kind=kind, window=side, fallback=1,
                                         **measured[start_from][n]))
                    else:
                        rows.append(dict(config=config, name=n, image_name=image_of[n], kind=kind, window=side, fallback=0,
                                         **case_metrics(folder/'label'/n, final)))
    if len(rows) != len(CONFIGS)*len(records):
        raise RuntimeError('Unexpected result table')
    save_rows(destination/'cases.csv', rows)
    finish(destination, signature, 'cases.csv', len(rows))
