"""One checkpoint on one cohort: the accepted single pass with the training sentence and with four other texts, and
the training sentence with one stochastic pass. Also the timing of one inference run for several numbers of passes."""
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import research_zoom.evaluate as accepted
from research_formal.metrics import case_metrics
from research_zoom_probe.evaluate import save_rows, done, finish
from train_detail_formal import TrainingPaused, set_random_seed, forward_batch
from .common import write_json, now, TEXTS, CONFIGS, TIMED


def draws_of(model, cfg, batch, passes):
    """[passes, B, H, W] probability maps of independent stochastic forward passes."""
    draws = torch.sigmoid(forward_batch(model, batch, cfg.MODEL.DEVICE, passes))
    if draws.dim() == 3 and passes == 1:
        draws = draws.unsqueeze(0)
    if draws.dim() != 4 or draws.shape[0] != passes:
        raise ValueError(f'Unexpected shape of the stochastic passes: {tuple(draws.shape)}')
    if not torch.isfinite(draws).all():
        raise ValueError('Non-finite prediction')
    return draws


def with_text(batch, text):
    return dict(batch, text_prompt=[text]*len(batch['mask_name']))


def texts_of(model, cfg, batch, prompt, protocol):
    """The strings of the five text conditions. The empty string is replaced by the fallback fixed in the protocol if
    the text encoder cannot process it; the decision is made on one batch and no accuracy is looked at."""
    texts = dict(fixed=prompt, **{k: protocol['prompts'][k] for k in TEXTS[1:]})
    note = 'empty string'
    try:
        with torch.no_grad():
            draws_of(model, cfg, with_text(batch, ''), 1)
    except Exception as error:      # noqa: BLE001 - any failure of the encoder on an empty text selects the fallback
        texts['empty'] = protocol['prompts']['empty_fallback']
        note = f'fallback, because the empty string failed: {type(error).__name__}: {error}'[:300]
    if len(set(texts.values())) != len(texts):
        raise ValueError('Two text conditions use the same string')
    return texts, note


def probe(model, cfg, folder, records, prompt, protocol, destination, signature, control, label):
    """-> cases.csv with one row per image and configuration (see common.CONFIGS)."""
    folder, destination = Path(folder), Path(destination)
    if done(destination, signature, 'cases.csv'):
        return
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'manifest.json', dict(signature=signature, complete=False, started_at=now()))
    state = dict(stage='', step=0)

    def tick():
        if control.pause_requested():
            control.update('paused', cohort=label)
            raise TrainingPaused()
        state['step'] += 1
        control.update('running', cohort=label, config=state['stage'], step=state['step'], total_steps=None)

    loader = DataLoader(accepted.dataset(folder, records), batch_size=protocol['batch_size'], shuffle=False, num_workers=0)
    torch.backends.cudnn.benchmark = True
    texts, note = texts_of(model, cfg, next(iter(loader)), prompt, protocol)
    write_json(destination/'texts.json', dict(texts=texts, empty_text=note))
    image_of, maps = {}, {config: {} for config in CONFIGS}
    for config in CONFIGS:
        state['stage'] = config
        text, passes = (prompt, 1) if config == 'single01' else (texts[config], protocol['mc_samples'])
        set_random_seed(cfg.seed)          # every condition starts from the same random stream (the accepted one)
        with torch.no_grad():
            for batch in loader:
                tick()
                if any(t != prompt for t in batch['text_prompt']):
                    raise ValueError('Unexpected prompt in the data')
                mean = draws_of(model, cfg, with_text(batch, text), passes).mean(0).float().cpu().numpy().astype(np.float32)
                for b, name in enumerate(batch['mask_name']):
                    maps[config][name] = mean[b]
                    image_of[name] = batch['image_name'][b]
    names = list(image_of)
    if len(names) != len(records) or any(len(maps[c]) != len(names) for c in CONFIGS):
        raise RuntimeError('Cohort coverage mismatch')
    rows = [dict(config=config, name=n, image_name=image_of[n],
                 **case_metrics(folder/'label'/n, np.ascontiguousarray(maps[config][n], dtype=np.float32)))
            for config in CONFIGS for n in names]
    save_rows(destination/'cases.csv', rows)
    finish(destination, signature, 'cases.csv', len(rows))


def timing(model, cfg, folder, records, prompt, protocol, path, control):
    """Wall-clock time of one inference run of the whole image at batch size 1, for each number of stochastic passes."""
    spec = protocol['timing']
    count = spec['warmup_images']+spec['images']
    loader = DataLoader(accepted.dataset(Path(folder), records[:count]), batch_size=1, shuffle=False, num_workers=0)
    batches = [batch for batch in loader]
    if len(batches) != count or any(b['text_prompt'][0] != prompt for b in batches):
        raise RuntimeError('Unexpected timing cohort')
    torch.backends.cudnn.benchmark = True
    result = {}
    for passes in TIMED:
        set_random_seed(cfg.seed)
        times = []
        with torch.no_grad():
            for index, batch in enumerate(batches):
                if control.pause_requested():
                    control.update('paused', cohort='timing')
                    raise TrainingPaused()
                control.update('running', cohort='timing', config=f'{passes} passes', step=index+1, total_steps=count)
                torch.cuda.synchronize()
                start = time.perf_counter()
                draws_of(model, cfg, batch, passes).mean(0)
                torch.cuda.synchronize()
                if index >= spec['warmup_images']:
                    times.append(1000*(time.perf_counter()-start))
        result[str(passes)] = dict(mean_ms=float(np.mean(times)), sd_ms=float(np.std(times, ddof=1)),
                                   median_ms=float(np.median(times)), n=len(times))
    write_json(path, dict(model=spec['model'], dataset=spec['dataset'], device=torch.cuda.get_device_name(0),
                          torch=torch.__version__, batch_size=1, passes=result, note=spec['note'], measured_at=now()))
