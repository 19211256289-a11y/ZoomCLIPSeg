"""First pass identical to the accepted evaluation; second pass on a rescaled window of the original pixels."""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

from datasets.dataloader import CLIP_NORMALIZE
from train_detail_formal import TrainingPaused, set_random_seed, forward_batch
from research.data import DualScaleDataset, DualScaleTransform
from research_formal.metrics import case_metrics, write_cases
from research_detail_test.evaluate import GenericPrompt
from .common import GENERIC, read, write_json, now, sha256
from .geometry import region, window_rule, crop, compose, rescale

KEYS = ('config', 'name', 'image_name', 'kind', 'window', 'fallback')


def tensors(raw):
    """Same preprocessing as DualScaleDataset(upsampled=False) + DualScaleTransform(False)."""
    convert = lambda a: CLIP_NORMALIZE(F.to_tensor(Image.fromarray(a.astype(np.uint8))))
    return convert(cv2.resize(raw, (224, 224))), convert(cv2.resize(raw, (448, 448)))


def dataset(folder, records):
    return GenericPrompt(DualScaleDataset(str(folder), 'Val', records, DualScaleTransform(False),
                                          image_size=224, detail_size=448, upsampled=False))


def check_preprocessing(folder, records):
    data = dataset(folder, records)
    for index in range(min(4, len(data))):
        sample = data[index]
        image, detail = tensors(cv2.imread(str(Path(folder)/'img'/sample['image_name'])))
        if not (torch.equal(image, sample['image']) and torch.equal(detail, sample['image_detail'])):
            raise RuntimeError('Second-pass preprocessing differs from the accepted first-pass preprocessing')


def build_cohort(condition, source, records, destination):
    """Materialises one synthetic cohort as lossless PNG. Returns (folder, records)."""
    source, destination = Path(source), Path(destination)
    if condition['scale'] == 1:
        return source, records
    manifest = destination/'cohort.json'
    if manifest.exists() and read(manifest).get('complete'):
        return destination, read(manifest)['records']
    for sub in ('img', 'label'):
        (destination/sub).mkdir(parents=True, exist_ok=True)
    made = []
    for index, record in enumerate(sorted(records, key=lambda r: r['Image'])):
        raw = cv2.imread(str(source/'img'/record['Image']))
        mask = cv2.imread(str(source/'label'/record['Ground Truth']), 0)
        if raw is None or mask is None:
            raise ValueError('Unreadable Val image or mask: '+record['Image'])
        if mask.shape != raw.shape[:2]:
            mask = cv2.resize(mask, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
        image, label = rescale(raw, (mask > 127).astype(np.uint8)*255, condition['scale'])
        name = f'{index:03d}.png'
        if not (cv2.imwrite(str(destination/'img'/name), image) and cv2.imwrite(str(destination/'label'/name), label)):
            raise OSError('PNG save failed')
        made.append({'Image': name, 'Ground Truth': name, 'Description': GENERIC, 'source_image': record['Image']})
    write_json(manifest, dict(complete=True, condition=condition, records=made, created_at=now()))
    return destination, made


def predict(model, cfg, batch):
    draws = torch.sigmoid(forward_batch(model, batch, cfg.MODEL.DEVICE, 30))
    if draws.shape[0] != 30:
        raise ValueError('Expected 30 MC samples')
    mean = draws.mean(0).cpu().numpy().astype(np.float32)
    if not np.isfinite(mean).all():
        raise ValueError('Non-finite prediction')
    return mean


def evaluate_condition(model, cfg, folder, records, protocol, destination, signature, control, label):
    """All configs of one checkpoint on one cohort. A paused job is rerun from its start with the same seeds."""
    folder, destination = Path(folder), Path(destination)
    manifest_path = destination/'manifest.json'
    if manifest_path.exists():
        previous = read(manifest_path)
        if previous.get('complete') and previous['signature'] == signature \
                and sha256(destination/'cases.csv') == previous['cases_sha256']:
            return
    destination.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, dict(signature=signature, complete=False, started_at=now()))
    loader = DataLoader(dataset(folder, records), batch_size=32, shuffle=False, num_workers=0)
    configs = protocol['configs']
    steps = len(loader)*(1+len(configs))

    def tick(step):
        if control.pause_requested():
            control.update('paused', cohort=label)
            raise TrainingPaused()
        control.update('running', cohort=label, step=step, total_steps=steps)

    set_random_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    first = {}
    with torch.no_grad():
        for index, batch in enumerate(loader):
            tick(index)
            if any(text != GENERIC for text in batch['text_prompt']):
                raise ValueError('Non-generic prompt reached the model')
            mean = predict(model, cfg, batch)
            for b, name in enumerate(batch['mask_name']):
                first[name] = (batch['image_name'][b], mean[b])
    if len(first) != len(records):
        raise RuntimeError('Cohort coverage mismatch')
    rows, single = [], {}
    for name, (image_name, probability) in first.items():
        single[name] = dict(config='single', name=name, image_name=image_name, kind='none', window=1., fallback=0,
                            **case_metrics(folder/'label'/name, probability))
        rows.append(single[name])
    for number, config in enumerate(configs):
        todo = []
        for name, (image_name, probability) in first.items():
            if config['oracle']:
                truth = cv2.imread(str(folder/'label'/name), 0)
                fraction, box = region(truth > 127)
            else:
                fraction, box = region(probability > .5)
            kind, side = window_rule(fraction, box, config['target_fraction'], protocol['zoom'], config['zoom_out'])
            if kind == 'none':
                rows.append(dict(single[name], config=config['name']))
            else:
                todo.append((name, image_name, kind, side, box))
        set_random_seed(cfg.seed)
        with torch.no_grad():
            for start in range(0, len(todo), 32):
                tick(len(loader)*(1+number)+start//32)
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
                    if final is None:
                        rows.append(dict(single[name], config=config['name'], kind=kind, window=side, fallback=1))
                    else:
                        rows.append(dict(config=config['name'], name=name, image_name=image_name, kind=kind, window=side,
                                         fallback=0, **case_metrics(folder/'label'/name, final)))
    if len(rows) != len(records)*(1+len(configs)) or any(list(r)[:6] != list(KEYS) for r in rows):
        raise RuntimeError('Unexpected result table')
    write_cases(destination/'cases.csv', rows)
    write_json(manifest_path, dict(signature=signature, complete=True, n=len(rows),
                                   cases_sha256=sha256(destination/'cases.csv'), finished_at=now(),
                                   device=torch.cuda.get_device_name()))
