"""Author-compatible mask metrics plus fixed-source size and omission analysis."""
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from utils.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance
from utils.main_utils import read_text


def foreground_threshold(cfg):
    areas = []
    text = Path(cfg.DATASET.TEXT_PROMPT_PATH) / 'Train_text.xlsx'
    for record in read_text(text):
        path = Path(cfg.DATASET.TRAIN_PATH) / 'label' / record['Ground Truth']
        mask = cv2.imread(str(path), 0)
        if mask is None:
            raise ValueError(f'Unreadable training mask: {path}')
        area = float((mask > 127).mean())
        if area:
            areas.append(area)
    if not areas:
        raise ValueError('Cannot define foreground-area threshold without positive training masks')
    return float(np.quantile(areas, .25))


def case_metrics(gt_path, probability):
    mask = cv2.imread(str(gt_path), 0)
    if mask is None:
        raise ValueError(f'Unreadable GT: {gt_path}')
    gt = mask > 127
    pred = cv2.resize((probability > .5).astype(np.uint8), (gt.shape[1], gt.shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    g, p, tp = int(gt.sum()), int(pred.sum()), int((gt & pred).sum())
    dsc = 1. if g+p == 0 else 2*tp/(g+p+1e-8)
    if not g or not p:
        nsd = 1. if not g and not p else 0.
    else:
        surface = compute_surface_distances(gt[..., None], pred[..., None], [1, 1, 1])
        nsd = float(compute_surface_dice_at_tolerance(surface, 2))
    low_gt = cv2.resize(gt.astype(np.uint8), probability.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    error = (probability-low_gt.astype(np.float32))**2
    return dict(DSC=round(dsc, 4), NSD=round(nsd, 4), gt_pixels=g, pred_pixels=p,
                tp_pixels=tp, fp_pixels=p-tp, fn_pixels=g-tp, gt_fraction=g/gt.size,
                foreground_recall=tp/g if g else None, foreground_precision=tp/p if p else None,
                brier_224=float(error.mean()), brier_fg_224=float(error[low_gt].mean()) if low_gt.any() else None,
                brier_bg_224=float(error[~low_gt].mean()) if (~low_gt).any() else None)


def average(rows, key, scale=1.):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values))*scale if values else None


def summarize(rows, threshold):
    positive = [r for r in rows if r['gt_pixels'] > 0]
    negative = [r for r in rows if r['gt_pixels'] == 0]
    small = [r for r in positive if r['gt_fraction'] <= threshold]
    tp, fp, fn = (sum(r[k] for r in rows) for k in ('tp_pixels', 'fp_pixels', 'fn_pixels'))
    quartiles = np.array_split(np.arange(len(positive)), 4)
    ordered = sorted(positive, key=lambda r: (r['gt_fraction'], r['name']))
    return dict(n=len(rows), n_positive=len(positive), n_negative=len(negative), n_small=len(small),
                DSC_percent=average(rows, 'DSC', 100), NSD_percent=average(rows, 'NSD', 100),
                positive_DSC_percent=average(positive, 'DSC', 100), small_DSC_percent=average(small, 'DSC', 100),
                micro_foreground_precision_percent=100*tp/(tp+fp) if tp+fp else None,
                micro_foreground_recall_percent=100*tp/(tp+fn) if tp+fn else None,
                empty_positive=sum(r['pred_pixels'] == 0 for r in positive),
                zero_overlap_positive=sum(r['tp_pixels'] == 0 for r in positive),
                recall_below_50=sum(r['foreground_recall'] < .5 for r in positive),
                precision_below_50=sum(r.get('foreground_precision') is not None and r['foreground_precision'] < .5 for r in rows),
                small_empty_positive=sum(r['pred_pixels'] == 0 for r in small),
                positive_DSC_below_10_percent=sum(r['DSC'] < .1 for r in positive),
                positive_DSC_below_50_percent=sum(r['DSC'] < .5 for r in positive),
                negative_false_positive=sum(r['pred_pixels'] > 0 for r in negative),
                brier_224=average(rows, 'brier_224'),
                local_area_quartile_DSC_percent=[average([ordered[i] for i in q], 'DSC', 100) for q in quartiles],
                source_train_area_threshold=threshold)


def write_cases(path, rows):
    path = Path(path)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
