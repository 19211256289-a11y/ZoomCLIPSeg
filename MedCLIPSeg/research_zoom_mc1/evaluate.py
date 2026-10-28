"""One checkpoint on one cohort: single pass with 30 and with 1 stochastic pass, and the locked two-pass rule (SCZ) with
1 stochastic pass in each run. The first pass is computed once with 30 passes exactly as in the accepted evaluation;
the 1-pass first map is the first of these 30 passes. The second pass of scz01 draws one new pass.
Also: end-to-end wall-clock time per image at batch size 1."""
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import research_zoom.evaluate as accepted
from research_zoom.geometry import region, window_rule, crop, compose
from research_formal.metrics import case_metrics
from research_zoom_probe.evaluate import save_rows, done, finish
from research_zoom_budget.evaluate import draws_of
from train_detail_formal import TrainingPaused, set_random_seed, forward_batch
from .common import write_json, now, CONFIGS


def one_draw(model, cfg, batch):
    """[1, B, H, W] probability map of one stochastic forward pass (the model is called with num_samples=1)."""
    draws = torch.sigmoid(forward_batch(model, batch, cfg.MODEL.DEVICE, 1))
    if draws.dim() == 3:
        draws = draws[None]
    if draws.dim() != 4 or draws.shape[0] != 1:
        raise ValueError(f'Unexpected shape of one stochastic pass: {tuple(draws.shape)}')
    if not torch.isfinite(draws).all():
        raise ValueError('Non-finite prediction')
    return draws


def as_numpy(maps):
    return maps.float().cpu().numpy().astype(np.float32)


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
    image_of, maps = {}, dict(single30={}, single01={})
    set_random_seed(cfg.seed)
    with torch.no_grad():
        for batch in loader:
            tick()
            if any(text != prompt for text in batch['text_prompt']):
                raise ValueError('Unexpected prompt reached the model')
            draws = draws_of(model, cfg, batch, 30)
            mean30, first = as_numpy(draws.mean(0)), as_numpy(draws[0])
            for b, name in enumerate(batch['mask_name']):
                maps['single30'][name], maps['single01'][name] = mean30[b], first[b]
                image_of[name] = batch['image_name'][b]
    names = list(image_of)
    if len(names) != len(records) or any(len(m) != len(names) for m in maps.values()):
        raise RuntimeError('Cohort coverage mismatch')

    measured, rows = {}, []
    for config in ('single30', 'single01'):
        measured[config] = {n: case_metrics(folder/'label'/n, np.ascontiguousarray(maps[config][n])) for n in names}
        rows += [dict(config=config, name=n, image_name=image_of[n], kind='none', window=1., fallback=0, **measured[config][n])
                 for n in names]

    state['stage'] = 'scz01'
    todo = []
    for n in names:
        fraction, box = region(maps['single01'][n] > .5)
        kind, side = window_rule(fraction, box, tau, zoom, False)
        if kind == 'none':
            rows.append(dict(config='scz01', name=n, image_name=image_of[n], kind='none', window=1., fallback=0,
                             **measured['single01'][n]))
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
            second = as_numpy(one_draw(model, cfg, batch)[0])
            for b, (n, kind, side, box) in enumerate(chunk):
                final = compose(second[b], kind, rects[b], shapes[b])
                if final is None:
                    rows.append(dict(config='scz01', name=n, image_name=image_of[n], kind=kind, window=side, fallback=1,
                                     **measured['single01'][n]))
                else:
                    rows.append(dict(config='scz01', name=n, image_name=image_of[n], kind=kind, window=side, fallback=0,
                                     **case_metrics(folder/'label'/n, final)))
    if len(rows) != len(CONFIGS)*len(records):
        raise RuntimeError('Unexpected result table')
    save_rows(destination/'cases.csv', rows)
    finish(destination, signature, 'cases.csv', len(rows))


def timing(model, cfg, folder, records, prompt, protocol, path, control):
    """Wall-clock time per image at batch size 1, from the preprocessed 224/448 tensors to the final probability map at
    image resolution, including the window rule, reading and cropping the original image, the second run and pasting.
    Four settings in one session: single pass and SCZ, each with 1 and with 30 stochastic passes per run."""
    spec = protocol['timing']
    tau, zoom = protocol['target_fraction'], protocol['zoom']
    folder = Path(folder)
    data = accepted.dataset(folder, records)
    count = min(len(data), spec['warmup_images']+spec['images'])
    if count <= spec['warmup_images']+10:
        raise ValueError('Not enough images for the timing run')
    samples = [data[i] for i in range(count)]
    if any(s['text_prompt'] != prompt for s in samples):
        raise ValueError('Unexpected prompt reached the model')
    torch.backends.cudnn.benchmark = True

    def run(sample, passes, zoomed):
        batch = dict(image=sample['image'][None], image_detail=sample['image_detail'][None], text_prompt=[prompt])
        draw = (lambda b: one_draw(model, cfg, b)) if passes == 1 else (lambda b: draws_of(model, cfg, b, passes))
        torch.cuda.synchronize()
        start = time.perf_counter()
        first = as_numpy(draw(batch).mean(0))[0]
        second_run = False
        if zoomed:
            fraction, box = region(first > .5)
            kind, side = window_rule(fraction, box, tau, zoom, False)
            if kind == 'in':
                raw = cv2.imread(str(folder/'img'/sample['image_name']))
                window, rect = crop(raw, kind, side, box)
                image, detail = accepted.tensors(window)
                second = as_numpy(draw(dict(image=image[None], image_detail=detail[None], text_prompt=[prompt])).mean(0))[0]
                compose(second, kind, rect, raw.shape[:2])
                second_run = True
            else:
                raw = cv2.imread(str(folder/'img'/sample['image_name']))
                cv2.resize(first, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            raw = cv2.imread(str(folder/'img'/sample['image_name']))
            cv2.resize(first, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR)
        torch.cuda.synchronize()
        return 1000*(time.perf_counter()-start), second_run

    result = {}
    settings = [('single01', 1, False), ('scz01', 1, True), ('single30', 30, False), ('t15', 30, True)]
    with torch.no_grad():
        for repetition in range(spec['repetitions']):
            for label, passes, zoomed in settings:
                set_random_seed(cfg.seed)
                times, second = [], []
                for index, sample in enumerate(samples):
                    if control.pause_requested():
                        control.update('paused', cohort='timing')
                        raise TrainingPaused()
                    control.update('running', cohort='timing', config=f'{label} (repetition {repetition+1})',
                                   step=index+1, total_steps=count)
                    ms, ran = run(sample, passes, zoomed)
                    if index >= spec['warmup_images']:
                        times.append(ms)
                        second.append(ran)
                result.setdefault(label, []).append(dict(mean_ms=float(np.mean(times)), sd_ms=float(np.std(times, ddof=1)),
                                                         median_ms=float(np.median(times)), n=len(times),
                                                         second_pass_share=float(np.mean(second))))
    summary = {label: dict(mean_ms_over_repetitions=float(np.mean([r['mean_ms'] for r in reps])),
                           range_ms=[float(min(r['mean_ms'] for r in reps)), float(max(r['mean_ms'] for r in reps))],
                           repetitions=reps) for label, reps in result.items()}
    write_json(path, dict(run=spec['run'], dataset=spec['dataset'], device=torch.cuda.get_device_name(0),
                          torch=torch.__version__, batch_size=1, warmup_images=spec['warmup_images'],
                          measured_images=count-spec['warmup_images'], settings=summary, note=spec['note'],
                          measured_at=now()))
