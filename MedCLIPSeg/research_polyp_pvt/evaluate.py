"""Inference as in the released test script (352 x 352 input, sum of both outputs, bilinear upsampling to the size of
the ground truth, sigmoid), scored with the metric code of this project. Differences from the released script, listed
in the protocol: threshold 0.5 on the sigmoid output without per-image min-max normalisation; last-epoch weights."""
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from research_formal.metrics import case_metrics
from .common import *
from .data import image_tensor
from .lib.pvt import PolypPVT


def load_model(seed, plan_digest):
    folder = training_dir(seed)
    manifest = read(folder/'manifest.json')
    if not manifest.get('complete') or manifest['plan_digest'] != plan_digest or sha256(folder/'final.pth') != manifest['sha256']:
        raise RuntimeError(f'Training of seed {seed} is not complete or its weights changed')
    model = PolypPVT(backbone_weights=WEIGHTS)
    model.load_state_dict(torch.load(folder/'final.pth', map_location='cpu'))
    return model.cuda().eval(), manifest['sha256']


def done(destination, signature):
    manifest = maybe(Path(destination)/'manifest.json')
    return bool(manifest.get('complete')) and manifest['signature'] == signature \
        and sha256(Path(destination)/'cases.csv') == manifest['sha256']


def evaluate(model, folder, records, size, warmup, destination, signature, control, label):
    """-> cases.csv with one row per image (keyed by the mask file name, as in the accepted evaluations)."""
    folder, destination = Path(folder), Path(destination)
    if done(destination, signature):
        return
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'manifest.json', dict(signature=signature, complete=False, started_at=now()))
    torch.backends.cudnn.benchmark = True
    rows, seconds = [], []
    with torch.no_grad():
        for index, record in enumerate(records):
            control.check(cohort=label)
            if index % 10 == 0:
                control.update('running', phase='evaluation', cohort=label, step=index+1, total_steps=len(records))
            truth = cv2.imread(str(folder/'label'/record['Ground Truth']), 0)
            if truth is None:
                raise ValueError(f"Unreadable ground truth: {record['Ground Truth']}")
            image = image_tensor(folder/'img'/record['Image'], size)[None].cuda()
            torch.cuda.synchronize()
            start = time.perf_counter()
            p1, p2 = model(image)
            probability = F.interpolate(p1+p2, size=truth.shape, mode='bilinear', align_corners=False).sigmoid()[0, 0]
            probability = probability.float().cpu().numpy()
            seconds.append(time.perf_counter()-start)
            if not np.isfinite(probability).all():
                raise RuntimeError('Non-finite prediction')
            rows.append(dict(name=record['Ground Truth'], image_name=record['Image'],
                             **case_metrics(folder/'label'/record['Ground Truth'], np.ascontiguousarray(probability))))
    if len(rows) != len(records) or len({r['name'] for r in rows}) != len(rows):
        raise RuntimeError('Cohort coverage mismatch')
    temporary = destination/'cases.csv.tmp'
    with open(temporary, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if v is None else v) for k, v in row.items()})
    temporary.replace(destination/'cases.csv')
    timed = np.array(seconds[warmup:]) if len(seconds) > warmup+1 else np.array(seconds)
    write_json(destination/'timing.json', dict(n=int(len(timed)), warmup=warmup, batch_size=1, passes=1,
                                               mean_ms=float(1000*timed.mean()), sd_ms=float(1000*timed.std(ddof=1)),
                                               median_ms=float(1000*np.median(timed)), device=torch.cuda.get_device_name()))
    write_json(destination/'manifest.json', dict(signature=signature, complete=True, n=len(rows), finished_at=now(),
                                                 sha256=sha256(destination/'cases.csv'), device=torch.cuda.get_device_name()))
