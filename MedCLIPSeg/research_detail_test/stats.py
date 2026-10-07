"""Paired image-level statistics. numpy only; no torch, no model access."""
import csv
import math
import numpy as np

FLOATS = ('DSC', 'NSD', 'gt_fraction', 'foreground_recall', 'foreground_precision')
INTS = ('gt_pixels', 'pred_pixels', 'tp_pixels', 'fp_pixels', 'fn_pixels')


def load_cases(path):
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as stream:
        for r in csv.DictReader(stream):
            row = dict(name=r['name'], image_name=r['image_name'])
            for k in INTS:
                row[k] = int(float(r[k]))
            for k in FLOATS:
                row[k] = float(r[k]) if r[k] not in ('', 'None') else None
            rows.append(row)
    if len({r['name'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate case names in {path}')
    return rows


def is_small(row, threshold):
    return row['gt_pixels'] > 0 and row['gt_fraction'] <= threshold


def arm_summary(rows, threshold):
    """Same definitions as research_formal.metrics.summarize."""
    positive = [r for r in rows if r['gt_pixels'] > 0]
    negative = [r for r in rows if r['gt_pixels'] == 0]
    small = [r for r in positive if r['gt_fraction'] <= threshold]
    tp, fp, fn = (sum(r[k] for r in rows) for k in ('tp_pixels', 'fp_pixels', 'fn_pixels'))
    mean = lambda rs, k: float(np.mean([r[k] for r in rs])) * 100 if rs else None
    return dict(n=len(rows), n_positive=len(positive), n_negative=len(negative), n_small=len(small),
                DSC_percent=mean(rows, 'DSC'), NSD_percent=mean(rows, 'NSD'),
                small_DSC_percent=mean(small, 'DSC'), small_NSD_percent=mean(small, 'NSD'),
                micro_foreground_precision_percent=100*tp/(tp+fp) if tp+fp else None,
                micro_foreground_recall_percent=100*tp/(tp+fn) if tp+fn else None,
                fp_pixels=fp, fn_pixels=fn,
                empty_positive=sum(r['pred_pixels'] == 0 for r in positive),
                zero_overlap_positive=sum(r['tp_pixels'] == 0 for r in positive),
                positive_DSC_below_50_percent=sum(r['DSC'] < .5 for r in positive),
                negative_false_positive=sum(r['pred_pixels'] > 0 for r in negative))


def _wilcoxon_p(values):
    d = values[values != 0]
    n = len(d)
    if n < 6:
        return None
    order = np.argsort(np.abs(d), kind='mergesort')
    absolute = np.abs(d)[order]
    ranks = np.empty(n)
    correction, i = 0., 0
    while i < n:
        j = i
        while j+1 < n and absolute[j+1] == absolute[i]:
            j += 1
        ranks[order[i:j+1]] = (i+j)/2+1
        t = j-i+1
        correction += t**3-t
        i = j+1
    w = ranks[d > 0].sum()
    mu = n*(n+1)/4
    sd = math.sqrt(n*(n+1)*(2*n+1)/24-correction/48)
    if sd == 0:
        return None
    z = (w-mu-.5*np.sign(w-mu))/sd
    return float(math.erfc(abs(z)/math.sqrt(2)))


def _permutation_p(values, draws, rng):
    observed = abs(values.mean())
    hits, done = 0, 0
    while done < draws:
        chunk = min(5000, draws-done)
        signs = rng.integers(0, 2, (chunk, len(values)), dtype=np.int8)*2-1
        hits += int((np.abs((signs*values).mean(1)) >= observed-1e-12).sum())
        done += chunk
    return (hits+1)/(draws+1)


def _describe(matrix, draws, boots, rng):
    """matrix: seeds x images paired differences in percentage points."""
    seeds, n = matrix.shape
    if n == 0:
        return dict(n=0)
    values = matrix.mean(0)
    order = np.sort(values)
    total = values.sum()
    k = min(2, n)
    trim = int(n*.1)
    resampled = values[rng.integers(0, n, (boots, n))].mean(1)
    seed_means = matrix.mean(1)
    result = dict(
        n=n, mean_pp=float(values.mean()), median_pp=float(np.median(values)),
        improved=int((values > 0).sum()), worsened=int((values < 0).sum()), tied=int((values == 0).sum()),
        per_seed_mean_pp=[float(v) for v in seed_means],
        per_seed_improved=[int((row > 0).sum()) for row in matrix],
        per_seed_worsened=[int((row < 0).sum()) for row in matrix],
        seeds_positive=int((seed_means > 0).sum()),
        bootstrap_ci95_pp=[float(np.quantile(resampled, .025)), float(np.quantile(resampled, .975))],
        permutation_p=float(_permutation_p(values, draws, rng)),
        wilcoxon_p=_wilcoxon_p(values),
        top2_gain_contribution_to_mean_pp=float(order[-k:].sum()/n),
        mean_of_others_without_top2_gains_pp=float((total-order[-k:].sum())/(n-k)) if n > k else None,
        mean_of_others_without_top2_losses_pp=float((total-order[:k].sum())/(n-k)) if n > k else None,
        trimmed_mean_10pct_pp=float(order[trim:n-trim].mean()) if n-2*trim > 0 else None)
    if seeds == 3:
        sd = float(np.std(seed_means, ddof=1))
        if sd > 0:
            t = float(seed_means.mean()/(sd/math.sqrt(3)))
            # Student t with 2 degrees of freedom has a closed-form CDF.
            result['seed_level_paired_t'] = dict(t=t, df=2, p=float(1-abs(t)/math.sqrt(2+t*t)))
    return result


def contrast(lhs, rhs, threshold, draws=100000, boots=10000, rng_seed=20261004, groups=None):
    """lhs/rhs: {seed: rows}. Returns paired statistics for lhs minus rhs."""
    seeds = sorted(lhs)
    if seeds != sorted(rhs):
        raise ValueError('Seeds differ between arms')
    names = sorted(r['name'] for r in lhs[seeds[0]])
    index = {}
    for arm in (lhs, rhs):
        for s in seeds:
            table = {r['name']: r for r in arm[s]}
            if sorted(table) != names:
                raise ValueError('Case lists differ between runs')
            index[id(arm), s] = table
    reference = index[id(rhs), seeds[0]]
    for arm in (lhs, rhs):
        for s in seeds:
            for name in names:
                if index[id(arm), s][name]['gt_pixels'] != reference[name]['gt_pixels']:
                    raise ValueError('Ground truth differs between runs for '+name)
    rng = np.random.default_rng(rng_seed)
    result = {}
    small = np.array([is_small(reference[name], threshold) for name in names])
    for metric in ('DSC', 'NSD'):
        matrix = np.array([[100*(index[id(lhs), s][n][metric]-index[id(rhs), s][n][metric]) for n in names] for s in seeds])
        result[metric] = dict(all=_describe(matrix, draws, boots, rng),
                              small=_describe(matrix[:, small], draws, boots, rng),
                              not_small=_describe(matrix[:, ~small], draws, boots, rng))
        if groups and metric == 'DSC':
            by_image = {reference[n]['image_name']: i for i, n in enumerate(names)}
            unique = np.array([[matrix[s, [by_image[i] for i in g]].mean() for g in groups] for s in range(len(seeds))])
            result['DSC']['unique_images'] = dict(n=len(groups), mean_pp=float(unique.mean()),
                                                  per_seed_mean_pp=[float(v) for v in unique.mean(1)])
    pixels = {}
    for key in ('fp_pixels', 'fn_pixels'):
        a = [sum(r[key] for r in lhs[s]) for s in seeds]
        b = [sum(r[key] for r in rhs[s]) for s in seeds]
        pixels[key+'_relative_change_percent'] = [100*(x-y)/y if y else None for x, y in zip(a, b)]
    for key in ('micro_foreground_precision_percent', 'micro_foreground_recall_percent',
                'positive_DSC_below_50_percent', 'zero_overlap_positive', 'empty_positive'):
        a = [arm_summary(lhs[s], threshold)[key] for s in seeds]
        b = [arm_summary(rhs[s], threshold)[key] for s in seeds]
        pixels[key+'_delta'] = [x-y if x is not None and y is not None else None for x, y in zip(a, b)]
    result['pixels_and_failures_per_seed'] = pixels
    result['seeds'] = seeds
    return result


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.
