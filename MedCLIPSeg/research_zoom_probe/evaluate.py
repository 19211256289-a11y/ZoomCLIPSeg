"""Inference of the probe. The first pass and the reference second pass repeat the accepted evaluation code path
(same preprocessing, same seeding, same batches); the variants differ only where their definition says so."""
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import research_zoom.evaluate as accepted
from research_zoom.geometry import region, compose
from research_formal.metrics import case_metrics
from train_detail_formal import TrainingPaused, set_random_seed
from .common import read, write_json, now, sha256
from .geometry import window_for, second_input


def save_rows(path, rows):
    keys = list(rows[0])
    if any(list(r) != keys for r in rows):
        raise RuntimeError('Rows with different columns')
    temporary = Path(str(path)+'.tmp')
    with open(temporary, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if v is None else v) for k, v in row.items()})
    temporary.replace(path)


def done(destination, signature, filename):
    """True if this job was completed with exactly these inputs and its file is unchanged."""
    manifest = Path(destination)/'manifest.json'
    if not manifest.exists():
        return False
    previous = read(manifest)
    return bool(previous.get('complete')) and previous['signature'] == signature \
        and sha256(Path(destination)/filename) == previous['sha256']


def finish(destination, signature, filename, n):
    write_json(Path(destination)/'manifest.json', dict(signature=signature, complete=True, n=n, finished_at=now(),
                                                       sha256=sha256(Path(destination)/filename),
                                                       device=torch.cuda.get_device_name()))


def first_pass(model, cfg, folder, records, prompt, batch_size, tick):
    """name -> (image file name, mean probability map on the 224 grid). Identical to the accepted first pass."""
    loader = DataLoader(accepted.dataset(folder, records), batch_size=batch_size, shuffle=False, num_workers=0)
    set_random_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    first = {}
    with torch.no_grad():
        for index, batch in enumerate(loader):
            tick(index, len(loader))
            if any(text != prompt for text in batch['text_prompt']):
                raise ValueError('Unexpected prompt reached the model')
            mean = accepted.predict(model, cfg, batch)
            for b, name in enumerate(batch['mask_name']):
                first[name] = (batch['image_name'][b], mean[b])
    if len(first) != len(records):
        raise RuntimeError('Cohort coverage mismatch')
    return first


def evaluate_variants(model, cfg, folder, records, variants, zoom, prompt, batch_size, destination, signature, control, label):
    """Single pass and every variant of one checkpoint on one cohort -> cases.csv."""
    folder, destination = Path(folder), Path(destination)
    if done(destination, signature, 'cases.csv'):
        return
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'manifest.json', dict(signature=signature, complete=False, started_at=now()))
    state = dict(stage='single')

    def tick(step, total):
        if control.pause_requested():
            control.update('paused', cohort=label)
            raise TrainingPaused()
        control.update('running', cohort=label, config=state['stage'], step=step, total_steps=total)

    first = first_pass(model, cfg, folder, records, prompt, batch_size, tick)
    rows, single = [], {}
    for name, (image_name, probability) in first.items():
        single[name] = dict(config='single', name=name, image_name=image_name, kind='none', window=1., fallback=0,
                            **case_metrics(folder/'label'/name, probability))
        rows.append(single[name])
    for variant in variants:
        state['stage'] = variant['name']
        todo = []
        for name, (image_name, probability) in first.items():
            fraction, box = region(probability > .5)
            kind, side = window_for(fraction, box, variant, zoom)
            if kind == 'none':
                rows.append(dict(single[name], config=variant['name']))
            else:
                todo.append((name, image_name, kind, side, box))
        set_random_seed(cfg.seed)
        with torch.no_grad():
            for start in range(0, len(todo), batch_size):
                tick(start//batch_size, (len(todo)+batch_size-1)//batch_size)
                chunk = todo[start:start+batch_size]
                images, details, rects, shapes = [], [], [], []
                for name, image_name, kind, side, box in chunk:
                    raw = cv2.imread(str(folder/'img'/image_name))
                    window, rect, shape = second_input(raw, kind, side, box, variant)
                    image, detail = accepted.tensors(window)
                    images.append(image)
                    details.append(detail)
                    rects.append(rect)
                    shapes.append(shape)
                mean = accepted.predict(model, cfg, dict(image=torch.stack(images), image_detail=torch.stack(details),
                                                         text_prompt=[prompt]*len(chunk)))
                for b, (name, image_name, kind, side, box) in enumerate(chunk):
                    final = compose(mean[b], kind, rects[b], shapes[b])
                    if final is None:
                        rows.append(dict(single[name], config=variant['name'], kind=kind, window=side, fallback=1))
                    else:
                        rows.append(dict(config=variant['name'], name=name, image_name=image_name, kind=kind, window=side,
                                         fallback=0, **case_metrics(folder/'label'/name, final)))
    if len(rows) != len(records)*(1+len(variants)):
        raise RuntimeError('Unexpected result table')
    save_rows(destination/'cases.csv', rows)
    finish(destination, signature, 'cases.csv', len(rows))


def image_scores(model, cfg, folder, records, prompt, batch_size, destination, signature, control, label):
    """Single pass of one checkpoint on one cohort -> scores.csv with the image-level score of every image."""
    folder, destination = Path(folder), Path(destination)
    if done(destination, signature, 'scores.csv'):
        return
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'manifest.json', dict(signature=signature, complete=False, started_at=now()))

    def tick(step, total):
        if control.pause_requested():
            control.update('paused', cohort=label)
            raise TrainingPaused()
        control.update('running', cohort=label, config='single', step=step, total_steps=total)

    first = first_pass(model, cfg, folder, records, prompt, batch_size, tick)
    rows = []
    for name, (image_name, probability) in first.items():
        metrics = case_metrics(folder/'label'/name, probability)
        rows.append(dict(name=name, image_name=image_name, max_probability=float(probability.max()),
                         predicted_fraction_224=float((probability > .5).mean()),
                         **{k: metrics[k] for k in ('DSC', 'NSD', 'gt_pixels', 'pred_pixels', 'tp_pixels', 'gt_fraction')}))
    save_rows(destination/'scores.csv', rows)
    finish(destination, signature, 'scores.csv', len(rows))


def timing(model, cfg, folder, records, variant, zoom, prompt, warmup, measured, control, label):
    """Wall-clock time per image at batch size 1: first pass, and second pass including cropping and pasting."""
    folder = Path(folder)
    data = accepted.dataset(folder, records)
    if len(data) < warmup+measured:
        raise ValueError('Not enough images for the timing run')
    set_random_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    rows = []
    with torch.no_grad():
        for index in range(warmup+measured):
            if control.pause_requested():
                control.update('paused', cohort=label)
                raise TrainingPaused()
            control.update('running', cohort=label, config='timing', step=index, total_steps=warmup+measured)
            sample = data[index]
            if sample['text_prompt'] != prompt:
                raise ValueError('Unexpected prompt reached the model')
            batch = dict(image=sample['image'][None], image_detail=sample['image_detail'][None], text_prompt=[prompt])
            torch.cuda.synchronize()
            start = time.perf_counter()
            mean = accepted.predict(model, cfg, batch)[0]
            torch.cuda.synchronize()
            first_seconds = time.perf_counter()-start
            fraction, box = region(mean > .5)
            kind, side = window_for(fraction, box, variant, zoom)
            second_seconds = forward_seconds = 0.
            if kind == 'in':
                torch.cuda.synchronize()
                start = time.perf_counter()
                raw = cv2.imread(str(folder/'img'/sample['image_name']))
                window, rect, shape = second_input(raw, kind, side, box, variant)
                image, detail = accepted.tensors(window)
                inner = time.perf_counter()
                second = accepted.predict(model, cfg, dict(image=image[None], image_detail=detail[None], text_prompt=[prompt]))[0]
                torch.cuda.synchronize()
                forward_seconds = time.perf_counter()-inner
                compose(second, kind, rect, shape)
                second_seconds = time.perf_counter()-start
            if index >= warmup:
                rows.append(dict(image=sample['image_name'], first=first_seconds, zoomed=kind == 'in',
                                 second=second_seconds, second_forward=forward_seconds))
    first = np.array([r['first'] for r in rows])
    second = np.array([r['second'] for r in rows if r['zoomed']])
    forward = np.array([r['second_forward'] for r in rows if r['zoomed']])
    share = float(np.mean([r['zoomed'] for r in rows]))
    total = np.array([r['first']+r['second'] for r in rows])
    ms = lambda a: dict(mean_ms=float(1000*a.mean()), sd_ms=float(1000*a.std(ddof=1)), median_ms=float(1000*np.median(a))) if len(a) > 1 else None
    return dict(n=len(rows), warmup=warmup, batch_size=1, mc_samples=30, device=torch.cuda.get_device_name(),
                first_pass=ms(first), second_pass_when_run=ms(second), second_pass_forward_only=ms(forward),
                second_pass_share=share, two_pass_total=ms(total),
                relative_cost_two_pass=float(total.mean()/first.mean()), measured_at=now())
