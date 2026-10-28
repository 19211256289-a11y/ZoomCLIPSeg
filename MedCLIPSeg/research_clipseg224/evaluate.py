"""Single pass and the locked scale-canonical zoom for CLIPSeg, scored with the metric code of this project."""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from research_formal.metrics import case_metrics
from research_polyp_pvt.data import MEAN, STD
from research_zoom.geometry import region, window_rule, crop, compose
from .common import *


def array_tensor(bgr, size):
    """Same preprocessing as research_polyp_pvt.data.image_tensor, from decoded pixels."""
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((size, size), Image.BILINEAR)
    array = (np.asarray(image, np.float32)/255-MEAN)/STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def predict(model, bgr, size):
    probability = torch.sigmoid(model(array_tensor(bgr, size)[None].cuda()))[0, 0].float().cpu().numpy()
    if not np.isfinite(probability).all():
        raise RuntimeError('Non-finite prediction')
    return np.ascontiguousarray(probability, dtype=np.float32)


def done(destination, signature):
    manifest = maybe(Path(destination)/'manifest.json')
    return bool(manifest.get('complete')) and manifest['signature'] == signature \
        and sha256(Path(destination)/'cases.csv') == manifest['sha256']


def evaluate(model, folder, records, protocol, destination, signature, control, label):
    """-> cases.csv with two rows per image: config 'single' and config 'scz'."""
    folder, destination = Path(folder), Path(destination)
    if done(destination, signature):
        return
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'manifest.json', dict(signature=signature, complete=False, started_at=now()))
    size, tau, zoom = protocol['recipe']['input_size'], protocol['target_fraction'], protocol['zoom']
    torch.backends.cudnn.benchmark = True
    rows = []
    with torch.no_grad():
        for index, record in enumerate(records):
            control.check(cohort=label)
            if index % 10 == 0:
                control.update('running', phase='evaluation', cohort=label, step=index+1, total_steps=len(records))
            name, image_name = record['Ground Truth'], record['Image']
            raw = cv2.imread(str(folder/'img'/image_name))
            if raw is None:
                raise ValueError(f'Unreadable image: {image_name}')
            first = predict(model, raw, size)
            single = dict(name=name, image_name=image_name, kind='none', window=1., fallback=0, **case_metrics(folder/'label'/name, first))
            rows.append(dict(config='single', **single))
            fraction, box = region(first > .5)
            kind, side = window_rule(fraction, box, tau, zoom, False)
            if kind == 'none':
                rows.append(dict(config='scz', **single))
                continue
            window, rect = crop(raw, kind, side, box)
            final = compose(predict(model, window, size), kind, rect, raw.shape[:2])
            if final is None:
                rows.append(dict(config='scz', **dict(single, kind=kind, window=side, fallback=1)))
            else:
                rows.append(dict(config='scz', name=name, image_name=image_name, kind=kind, window=side, fallback=0,
                                 **case_metrics(folder/'label'/name, final)))
    if len(rows) != 2*len(records) or len({(r['config'], r['name']) for r in rows}) != len(rows):
        raise RuntimeError('Cohort coverage mismatch')
    temporary = destination/'cases.csv.tmp'
    with open(temporary, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if v is None else v) for k, v in row.items()})
    temporary.replace(destination/'cases.csv')
    write_json(destination/'manifest.json', dict(signature=signature, complete=True, n=len(rows), finished_at=now(),
                                                 sha256=sha256(destination/'cases.csv'), device=torch.cuda.get_device_name()))
