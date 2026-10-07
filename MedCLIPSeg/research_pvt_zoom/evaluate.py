"""Single pass (the accepted Polyp-PVT inference) and the locked two-pass zoom for one model on one cohort."""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from research_formal.metrics import case_metrics
from research_polyp_pvt.data import MEAN, STD, decode
from research_zoom.geometry import region, window_rule, crop, compose
from .common import write_json, now, sha256, maybe


def tensor(pixels, size):
    """The preprocessing of research_polyp_pvt.data.image_tensor for an image that is already decoded (BGR)."""
    image = Image.fromarray(cv2.cvtColor(np.ascontiguousarray(pixels), cv2.COLOR_BGR2RGB)).resize((size, size), Image.BILINEAR)
    array = (np.asarray(image, np.float32)/255-MEAN)/STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def logits(model, pixels, size):
    p1, p2 = model(tensor(pixels, size)[None].cuda())
    return p1+p2


def done(destination, signature):
    manifest = maybe(Path(destination)/'manifest.json')
    return bool(manifest.get('complete')) and manifest['signature'] == signature \
        and sha256(Path(destination)/'cases.csv') == manifest['sha256']


def evaluate(model, folder, records, protocol, destination, signature, control, label):
    """-> cases.csv with two rows per image: config 'single' and config 't15'."""
    folder, destination = Path(folder), Path(destination)
    if done(destination, signature):
        return
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'manifest.json', dict(signature=signature, complete=False, started_at=now()))
    size, tau, zoom = protocol['input_size'], protocol['target_fraction'], protocol['zoom']
    torch.backends.cudnn.benchmark = True
    rows = []
    with torch.no_grad():
        for index, record in enumerate(records):
            control.check(cohort=label)
            if index % 10 == 0:
                control.update('running', phase='evaluation', cohort=label, step=index+1, total_steps=len(records))
            label_path = folder/'label'/record['Ground Truth']
            truth = decode(label_path, 0)
            raw = decode(folder/'img'/record['Image'])
            out = logits(model, raw, size)
            grid = out.sigmoid()[0, 0].float().cpu().numpy()
            first = F.interpolate(out, size=truth.shape, mode='bilinear', align_corners=False).sigmoid()[0, 0].float().cpu().numpy()
            first = np.ascontiguousarray(first)
            common = dict(name=record['Ground Truth'], image_name=record['Image'])
            single = case_metrics(label_path, first)
            rows.append(dict(config='single', **common, kind='none', window=1., fallback=0, **single))
            fraction, box = region(grid > .5)
            kind, side = window_rule(fraction, box, tau, zoom, False)
            if kind == 'none':
                rows.append(dict(config='t15', **common, kind='none', window=1., fallback=0, **single))
                continue
            window, rect = crop(raw, kind, side, box)
            second = logits(model, window, size).sigmoid()[0, 0].float().cpu().numpy()
            final = compose(second, kind, rect, raw.shape[:2])
            if final is None:
                rows.append(dict(config='t15', **common, kind=kind, window=side, fallback=1, **single))
            else:
                rows.append(dict(config='t15', **common, kind=kind, window=side, fallback=0,
                                 **case_metrics(label_path, np.ascontiguousarray(final))))
    if len(rows) != 2*len(records):
        raise RuntimeError('Unexpected result table')
    temporary = destination/'cases.csv.tmp'
    with open(temporary, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if v is None else v) for k, v in row.items()})
    temporary.replace(destination/'cases.csv')
    write_json(destination/'manifest.json', dict(signature=signature, complete=True, n=len(rows), finished_at=now(),
                                                 sha256=sha256(destination/'cases.csv'), device=torch.cuda.get_device_name()))
