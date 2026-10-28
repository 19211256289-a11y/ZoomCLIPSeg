"""One checkpoint on one cohort: the single pass and the locked two-pass rule (SCZ) with the binarisation threshold t
applied everywhere (the window rule, the empty-second-pass fallback and the final mask), for t in THRESHOLDS.
The first pass is computed once with 30 stochastic passes exactly as in the accepted evaluation; the second pass draws
30 new passes for every threshold whose window differs, so that t = 0.5 repeats the accepted method."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import research_zoom.evaluate as accepted
from research_zoom.geometry import region, window_rule, crop
from research_formal.metrics import case_metrics
from research_zoom_probe.evaluate import save_rows, done, finish
from research_zoom_budget.evaluate import draws_of
from train_detail_formal import TrainingPaused, set_random_seed
from .common import write_json, now, CONFIGS, THRESHOLDS, TAG

BRIER = ('brier_224', 'brier_fg_224', 'brier_bg_224')


def as_numpy(maps):
    return maps.float().cpu().numpy().astype(np.float32)


def metrics_at(gt_path, probability, t):
    """case_metrics binarises at 0.5; shifting the map by 0.5 - t binarises it at t. The Brier scores of a shifted map
    are meaningless and are left empty."""
    shifted = np.clip(probability.astype(np.float32)-np.float32(t)+np.float32(.5), 0., 1.)
    row = case_metrics(gt_path, np.ascontiguousarray(shifted, dtype=np.float32))
    row.update({k: None for k in BRIER})
    return row


def compose_at(second, kind, rect, shape, t):
    """As research_zoom.geometry.compose (zoom-in only), with the emptiness test at threshold t."""
    if kind != 'in':
        raise ValueError(kind)
    H, W = shape
    x0, y0, w, h = rect
    patch = cv2.resize(second.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    full = np.zeros((H, W), np.float32)
    full[y0:y0+h, x0:x0+w] = patch
    return full if (full > t).any() else None


def evaluate(model, cfg, folder, records, prompt, protocol, destination, signature, control, label):
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
    image_of, first = {}, {}
    set_random_seed(cfg.seed)
    with torch.no_grad():
        for batch in loader:
            tick()
            if any(text != prompt for text in batch['text_prompt']):
                raise ValueError('Unexpected prompt reached the model')
            mean = as_numpy(draws_of(model, cfg, batch, 30).mean(0))
            for b, name in enumerate(batch['mask_name']):
                first[name] = mean[b]
                image_of[name] = batch['image_name'][b]
    names = list(image_of)
    if len(names) != len(records):
        raise RuntimeError('Cohort coverage mismatch')

    rows = []
    for t in THRESHOLDS:
        single = {n: metrics_at(folder/'label'/n, first[n], t) for n in names}
        rows += [dict(config=f'single_t{TAG[t]}', name=n, image_name=image_of[n], kind='none', window=1., fallback=0, **single[n])
                 for n in names]
        config = f'scz_t{TAG[t]}'
        state['stage'] = config
        todo = []
        for n in names:
            fraction, box = region(first[n] > t)
            kind, side = window_rule(fraction, box, tau, zoom, False)
            if kind == 'none':
                rows.append(dict(config=config, name=n, image_name=image_of[n], kind='none', window=1., fallback=0, **single[n]))
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
                second = as_numpy(draws_of(model, cfg, batch, 30).mean(0))
                for b, (n, kind, side, box) in enumerate(chunk):
                    final = compose_at(second[b], kind, rects[b], shapes[b], t)
                    if final is None:
                        rows.append(dict(config=config, name=n, image_name=image_of[n], kind=kind, window=side, fallback=1,
                                         **single[n]))
                    else:
                        rows.append(dict(config=config, name=n, image_name=image_of[n], kind=kind, window=side, fallback=0,
                                         **metrics_at(folder/'label'/n, final, t)))
    if len(rows) != len(CONFIGS)*len(records):
        raise RuntimeError('Unexpected result table')
    save_rows(destination/'cases.csv', rows)
    finish(destination, signature, 'cases.csv', len(rows))
