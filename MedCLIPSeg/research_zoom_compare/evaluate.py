"""Single pass plus the generic alternatives of one checkpoint on one cohort. Tiles use the second-pass preprocessing."""
from pathlib import Path

import cv2
import numpy as np
import torch

import research_zoom.evaluate as accepted
from research_formal.metrics import case_metrics
from research_zoom_probe.evaluate import save_rows, done, finish, first_pass
from train_detail_formal import TrainingPaused, set_random_seed
from .common import write_json, now
from .tiles import tile_rects, stitch, full_resolution

LAYOUTS = ('tile2', 'tile15')


def compare(model, cfg, folder, records, prompt, batch_size, images_per_batch, destination, signature, control, label):
    """-> cases.csv with one row per image and configuration (single, hflip, tile2, ms12, tile15, ms123)."""
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
    rows = [dict(config='single', name=name, image_name=image_name, **case_metrics(folder/'label'/name, probability))
            for name, (image_name, probability) in first.items()]
    state['stage'] = 'alternatives'
    names = list(first)
    set_random_seed(cfg.seed)
    with torch.no_grad():
        for start in range(0, len(names), images_per_batch):
            tick(start//images_per_batch, (len(names)+images_per_batch-1)//images_per_batch)
            chunk = names[start:start+images_per_batch]
            images, details, plan = [], [], []
            for name in chunk:
                raw = cv2.imread(str(folder/'img'/first[name][0]))
                H, W = raw.shape[:2]
                windows = [np.ascontiguousarray(raw[:, ::-1])]
                rects = {layout: tile_rects(H, W, layout) for layout in LAYOUTS}
                for layout in LAYOUTS:
                    windows += [raw[y0:y0+h, x0:x0+w] for x0, y0, w, h in rects[layout]]
                for window in windows:
                    image, detail = accepted.tensors(window)
                    images.append(image)
                    details.append(detail)
                plan.append((name, (H, W), rects))
            if len(images) > batch_size:
                raise RuntimeError('Batch larger than the protocol allows')
            mean = accepted.predict(model, cfg, dict(image=torch.stack(images), image_detail=torch.stack(details),
                                                     text_prompt=[prompt]*len(images)))
            cursor = 0
            for name, shape, rects in plan:
                image_name, single = first[name]
                maps = {'hflip': (single+mean[cursor][:, ::-1])/2}
                cursor += 1
                for layout in LAYOUTS:
                    maps[layout] = stitch(mean[cursor:cursor+4], rects[layout], shape)
                    cursor += 4
                full = full_resolution(single, shape)
                maps['ms12'] = (full+maps['tile2'])/2
                maps['ms123'] = (full+maps['tile15']+maps['tile2'])/3
                for config in ('hflip', 'tile2', 'ms12', 'tile15', 'ms123'):
                    rows.append(dict(config=config, name=name, image_name=image_name,
                                     **case_metrics(folder/'label'/name, np.ascontiguousarray(maps[config], dtype=np.float32))))
            if cursor != len(images):
                raise RuntimeError('Window bookkeeping error')
    if len(rows) != 6*len(records):
        raise RuntimeError('Unexpected result table')
    save_rows(destination/'cases.csv', rows)
    finish(destination, signature, 'cases.csv', len(rows))
