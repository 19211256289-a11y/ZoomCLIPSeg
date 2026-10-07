"""Re-runs the accepted two-pass evaluation of one checkpoint with identical seeds and batching, and saves the masks.

Purpose: qualitative figures. The saved scores must reproduce the accepted per-image records; otherwise the run stops.
Also writes the lesion area fractions of the Kvasir training masks (CPU only) for a size-distribution figure.
"""
from local_runtime import configure_local_models
configure_local_models()
import argparse
import csv
import gc
import os
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from research_zoom_test.common import (ROOT, WORKSPACE, GENERIC, TARGETS, OUTPUT as ACCEPTED, load_protocol, read, maybe,
                                       now, write_json, accepted_runs, load_test_protocol)

RUN = 'detail_seed666'
NAME = 'zoom_export_v1'
CONTROL = WORKSPACE/'control/research_zoom_export'
OUT = ROOT/'outputs_research'/NAME/RUN
TOLERANCE = 0.02   # largest allowed per-image DSC difference to the accepted record (absolute, on the 0-1 scale)


def train_fractions():
    from research_zoom_ext.common import lesion_fractions
    from research_detail_formal import common as formal
    values = lesion_fractions(formal.make_config(formal.load_suite(), 'baseline_seed666'))
    write_json(ROOT/'outputs_research'/NAME/'kvasir_train_lesion_fractions.json', dict(n=len(values), fractions=[float(v) for v in values]))


def export_target(model, cfg, protocol, target, control):
    """Same order of operations and random seeds as research_zoom.evaluate.evaluate_condition, plus saving."""
    import torch
    from torch.utils.data import DataLoader
    from train_detail_formal import TrainingPaused, set_random_seed
    from utils.main_utils import read_text
    from research_formal.metrics import case_metrics
    from research_zoom.evaluate import dataset, predict, tensors
    from research_zoom.geometry import region, window_rule, crop, compose
    folder = ROOT/'data'/target/'Test_Folder'
    records = read_text(ROOT/'data'/target/'Prompts_Folder'/protocol['test_list'])
    destination = OUT/target
    if maybe(destination/'manifest.json').get('complete'):
        return
    for sub in ('first', 'final'):
        (destination/sub).mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset(folder, records), batch_size=32, shuffle=False, num_workers=0)
    config = protocol['config']

    def tick(step, total):
        if control.pause_requested():
            control.update('paused', cohort=target)
            raise TrainingPaused()
        control.update('running', cohort=target, step=step, total_steps=total)

    set_random_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    first = {}
    with torch.no_grad():
        for index, batch in enumerate(loader):
            tick(index, 2*len(loader))
            mean = predict(model, cfg, batch)
            for b, name in enumerate(batch['mask_name']):
                first[name] = (batch['image_name'][b], mean[b])
    rows, todo = {}, []
    for number, (name, (image_name, probability)) in enumerate(first.items()):
        key = f'{number:06d}'
        raw = cv2.imread(str(folder/'img'/image_name))
        height, width = raw.shape[:2]
        single = cv2.resize((probability > .5).astype(np.uint8)*255, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(destination/'first'/f'{key}.png'), single)
        metrics = case_metrics(folder/'label'/name, probability)
        rows[name] = dict(key=key, name=name, image_name=image_name, width=width, height=height, kind='none', x0='', y0='', w='', h='',
                          fallback=0, DSC_single=metrics['DSC'], DSC_two=metrics['DSC'], gt_fraction=metrics['gt_fraction'])
        cv2.imwrite(str(destination/'final'/f'{key}.png'), single)
        fraction, box = region(probability > .5)
        kind, side = window_rule(fraction, box, config['target_fraction'], protocol['zoom'], config['zoom_out'])
        if kind != 'none':
            todo.append((name, image_name, kind, side, box))
    set_random_seed(cfg.seed)
    with torch.no_grad():
        for start in range(0, len(todo), 32):
            tick(len(loader) + start//32, 2*len(loader))
            chunk = todo[start:start+32]
            images, details, rects, shapes = [], [], [], []
            for name, image_name, kind, side, box in chunk:
                raw = cv2.imread(str(folder/'img'/image_name))
                window, rect = crop(raw, kind, side, box)
                image, detail = tensors(window)
                images.append(image)
                details.append(detail)
                rects.append(rect)
                shapes.append(raw.shape[:2])
            mean = predict(model, cfg, dict(image=torch.stack(images), image_detail=torch.stack(details),
                                            text_prompt=[GENERIC]*len(chunk)))
            for b, (name, image_name, kind, side, box) in enumerate(chunk):
                final = compose(mean[b], kind, rects[b], shapes[b])
                row = rows[name]
                row.update(kind=kind, x0=rects[b][0], y0=rects[b][1], w=rects[b][2], h=rects[b][3])
                if final is None:
                    row['fallback'] = 1
                    continue
                row['DSC_two'] = case_metrics(folder/'label'/name, final)['DSC']
                cv2.imwrite(str(destination/'final'/f"{row['key']}.png"), (final > .5).astype(np.uint8)*255)
    # the saved masks must correspond to the accepted evaluation
    accepted = {}
    with open(ACCEPTED/'evaluations'/RUN/target/'cases.csv', newline='', encoding='utf-8-sig') as stream:
        for r in csv.DictReader(stream):
            accepted[r['config'], r['name']] = float(r['DSC'])
    worst = max(max(abs(r['DSC_single']-accepted['single', n]), abs(r['DSC_two']-accepted['t15', n])) for n, r in rows.items())
    with open(destination/'cases.csv', 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(next(iter(rows.values()))))
        writer.writeheader()
        writer.writerows(rows.values())
    write_json(destination/'manifest.json', dict(complete=worst <= TOLERANCE, n=len(rows), largest_case_difference=worst,
                                                 tolerance=TOLERANCE, finished_at=now()))
    if worst > TOLERANCE:
        raise RuntimeError(f'{target}: exported predictions differ from the accepted records (largest DSC difference {worst:.4f})')
    print('EXPORTED', target, len(rows), 'images; largest DSC difference to the accepted records', round(worst, 4), flush=True)


def status():
    queue = maybe(CONTROL/'queue.json')
    print('Mask export:', queue.get('state', 'not_started'), '| pause requested:', (CONTROL/'PAUSE.request').exists())
    if queue.get('state') == 'waiting_for_gpu':
        print('  waiting: another queue holds the GPU; the export starts by itself when it is free')
    print('  targets done:', [t for t in TARGETS if maybe(OUT/t/'manifest.json').get('complete')])
    if queue.get('error'):
        print('  ERROR:', queue['error'])
    print('  updated:', queue.get('updated_at', 'n/a'), '(UTC)')


def work():
    import torch
    from local_reproduce import acquire_gpu_queue
    from train_detail_formal import TrainingControl, TrainingPaused
    from research_detail_test.evaluate import load_model
    CONTROL.mkdir(parents=True, exist_ok=True)
    pause, path = CONTROL/'PAUSE.request', CONTROL/'queue.json'
    pause.unlink(missing_ok=True)
    queue = maybe(path) or {}

    def save(**fields):
        queue.update(fields, updated_at=now(), pid=os.getpid())
        write_json(path, queue)

    save(state='preparing', error=None)
    try:
        train_fractions()
    except Exception as exc:   # the size-distribution file is optional
        print('Training lesion fractions were not written:', exc, flush=True)
    while True:
        try:
            lock = acquire_gpu_queue()
            break
        except RuntimeError:
            save(state='waiting_for_gpu')
            if pause.exists():
                save(state='paused')
                return 75
            time.sleep(60)
    with lock:
        save(state='running')
        try:
            protocol = load_protocol()
            digest = accepted_runs(load_test_protocol())[RUN]['checkpoint_sha256']
            model, cfg, _ = load_model(RUN, digest)
            control = TrainingControl(str(CONTROL), run_id=RUN, phase='export')
            for target in TARGETS:
                export_target(model, cfg, protocol, target, control)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            save(state='completed')
            print('Export complete:', OUT, flush=True)
            return 0
        except TrainingPaused:
            save(state='paused')
            return 75
        except BaseException as exc:
            save(state='failed', error=f'{type(exc).__name__}: {exc}')
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['run', 'pause', 'status'], default='status')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'status':
        status()
        return 0
    if args.action == 'pause':
        CONTROL.mkdir(parents=True, exist_ok=True)
        (CONTROL/'PAUSE.request').write_text(now(), encoding='utf-8')
        print('Pause requested.')
        return 0
    return work()


if __name__ == '__main__':
    raise SystemExit(main())
