"""Analysis of the probe from its saved per-image files. No model is involved; can be rerun at any time."""
import csv

import numpy as np

from .common import *


def rows_of(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def stats(diff, rng, draws, perms):
    """Mean, 95% bootstrap interval over images and two-sided sign-flip permutation p of per-image differences (pp)."""
    diff = np.asarray(diff, float)
    boot = diff[rng.integers(0, len(diff), (draws, len(diff)))].mean(1)
    null = np.abs((rng.choice([-1., 1.], (perms, len(diff)))*diff).mean(1))
    return dict(mean_pp=float(diff.mean()), ci95_pp=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                p=float((1+np.sum(null >= abs(diff.mean())-1e-12))/(perms+1)), n=int(len(diff)))


def macro_stats(per_set, rng, draws, perms):
    """Equal-weight mean over datasets; images are resampled and sign-flipped within each dataset."""
    values = [np.asarray(v, float) for v in per_set.values()]
    observed = float(np.mean([v.mean() for v in values]))
    boot = np.mean([v[rng.integers(0, len(v), (draws, len(v)))].mean(1) for v in values], 0)
    null = np.mean([(rng.choice([-1., 1.], (perms, len(v)))*v).mean(1) for v in values], 0)
    return dict(mean_pp=observed, ci95_pp=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                p=float((1+np.sum(np.abs(null) >= abs(observed)-1e-12))/(perms+1)))


# ---------------------------------------------------------------- Part A
def analyse_polyps(protocol, rng, draws, perms):
    from research_detail_test.common import TARGETS
    polyp = protocol['polyp']
    targets, ood = list(TARGETS), list(TARGETS)[1:]
    table = {}                                          # (config, target, seed, image) -> DSC in percent
    zoomed = {}
    for run in polyp['runs']:
        for target in targets:
            for r in rows_of(OUTPUT/'evaluations'/'zoom'/run/target/'cases.csv'):
                table[(r['config'], target, seed_of(run), r['name'])] = 100*float(r['DSC'])
                zoomed[(r['config'], target, seed_of(run), r['name'])] = r['kind'] == 'in'
    names = {t: sorted({n for (c, tt, s, n) in table if c == 'single' and tt == t}) for t in targets}
    result = {}
    for variant in polyp['all_variants']:
        name = variant['name']
        seeds = polyp['groups']['resolution'] if variant['group'] == 'reference' else polyp['groups'][variant['group']]
        block = dict(group=variant['group'], seeds=seeds,
                     constants={k: variant[k] for k in ('tau', 'margin', 'min_window', 'trigger', 'pixels', 'output')})
        for scope, use in (('all_seeds', seeds), ('first_three', [s for s in seeds if s in polyp['groups']['sensitivity']])):
            per_set = {t: [float(np.mean([table[(name, t, s, n)]-table[('single', t, s, n)] for s in use])) for n in names[t]]
                       for t in targets}
            per_seed = [float(np.mean([np.mean([table[(name, t, s, n)]-table[('single', t, s, n)] for n in names[t]]) for t in ood]))
                        for s in use]
            block[scope] = dict(
                seeds=use, sets={t: stats(per_set[t], rng, draws, perms) for t in targets},
                ood_macro=macro_stats({t: per_set[t] for t in ood}, rng, draws, perms), ood_macro_per_seed_pp=per_seed,
                seeds_positive=int(sum(v > 0 for v in per_seed)),
                single_ood=float(np.mean([np.mean([table[('single', t, s, n)] for s in use for n in names[t]]) for t in ood])),
                two_pass_ood=float(np.mean([np.mean([table[(name, t, s, n)] for s in use for n in names[t]]) for t in ood])),
                second_pass_share_ood=float(np.mean([np.mean([zoomed[(name, t, s, n)] for s in use for n in names[t]]) for t in ood])))
        result[name] = block
    reference = result['t15']
    resolution = {}
    for name in ('in224', 'in224_out224'):
        seeds = polyp['groups']['resolution']
        per_set = {t: [float(np.mean([table[('t15', t, s, n)]-table[(name, t, s, n)] for s in seeds])) for n in names[t]] for t in ood}
        gain, full = result[name]['all_seeds']['ood_macro']['mean_pp'], reference['all_seeds']['ood_macro']['mean_pp']
        resolution[name] = dict(gain_pp=gain, reference_gain_pp=full, ratio=float(gain/full) if full else None,
                                t15_minus_variant=macro_stats(per_set, rng, draws, perms))
    rule = polyp['reading_rules_fixed_in_advance']['resolution']
    ratio = resolution['in224']['ratio']
    verdict = ('scale_explains_most' if ratio >= rule['scale_explains_most_if_ratio_at_least']
               else 'original_pixels_explain_most' if ratio <= rule['original_pixels_explain_most_if_ratio_at_most'] else 'mixed')
    sensitivity = [n for n, b in result.items() if b['group'] == 'sensitivity']
    minimum = polyp['reading_rules_fixed_in_advance']['sensitivity']['min_gain_pp']
    robust = all(result[n]['all_seeds']['ood_macro']['mean_pp'] >= minimum and
                 result[n]['all_seeds']['seeds_positive'] == len(result[n]['all_seeds']['seeds']) for n in sensitivity)
    return dict(variants=result, resolution=dict(resolution, verdict=verdict), sensitivity_robust=bool(robust),
                reference_first_three_gain_pp=reference['first_three']['ood_macro']['mean_pp'])


# ---------------------------------------------------------------- Part B
def gated(rows, theta):
    """DSC in percent of every image when predictions whose image score does not exceed theta are made empty."""
    out = []
    for r in rows:
        if r['score'] > theta:
            out.append(r['dsc'])
        else:
            out.append(100. if r['gt'] == 0 else 0.)
    return np.array(out)


def specificity(rows, theta):
    normal = [r for r in rows if r['gt'] == 0]
    return float(np.mean([r['score'] <= theta for r in normal])) if normal else None


def sensitivity(rows, theta):
    lesion = [r for r in rows if r['gt'] > 0]
    return float(np.mean([r['score'] > theta for r in lesion])) if lesion else None


def calibrate(detail_normals, target):
    """Smallest theta >= 0.5 at which the specificity on these lesion-free images reaches the target."""
    if not detail_normals:
        return .5
    for theta in sorted({.5}|{r['score'] for r in detail_normals if r['score'] > .5}):
        if np.mean([r['score'] <= theta for r in detail_normals]) >= target-1e-12:
            return float(theta)
    raise RuntimeError('unreachable')


def auc(rows):
    pos = np.array([r['score'] for r in rows if r['gt'] > 0])
    neg = np.array([r['score'] for r in rows if r['gt'] == 0])
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None, :]).sum()+.5*(pos[:, None] == neg[None, :]).sum())/(len(pos)*len(neg)))


def analyse_ultrasound(protocol, rng, draws, perms):
    ultra = protocol['ultrasound']
    test, ood = ultra['cohorts'][1], ultra['cohorts'][2:]
    data = {}
    for run in ultra['runs']:
        for cohort in ultra['cohorts']:
            data[(run, cohort)] = [dict(name=r['name'], score=float(r['max_probability']), dsc=100*float(r['DSC']),
                                        gt=int(float(r['gt_pixels'])), pred=int(float(r['pred_pixels'])))
                                   for r in sorted(rows_of(OUTPUT/'evaluations'/'detect'/run/cohort/'scores.csv'), key=lambda r: r['name'])]
    per_seed = []
    averaged = {mode: {c: [] for c in ood} for mode in ('ungated', 'val_gate', 'test_gate')}
    for seed in ultra['seeds']:
        b, d = f'baseline_seed{seed}', f'detail_seed{seed}'
        entry = dict(seed=seed, n_val_lesion_free=sum(r['gt'] == 0 for r in data[(b, 'Val')]),
                     n_test_lesion_free=sum(r['gt'] == 0 for r in data[(b, test)]))
        thetas = dict(ungated=.5,
                      val_gate=calibrate([r for r in data[(d, 'Val')] if r['gt'] == 0], specificity(data[(b, 'Val')], .5) or 0.),
                      test_gate=calibrate([r for r in data[(d, test)] if r['gt'] == 0], specificity(data[(b, test)], .5) or 0.))
        entry['baseline'] = dict(val_specificity=specificity(data[(b, 'Val')], .5), test_specificity=specificity(data[(b, test)], .5),
                                 test_sensitivity=sensitivity(data[(b, test)], .5), test_auc=auc(data[(b, test)]),
                                 ood_empty=int(sum(r['score'] <= .5 for c in ood for r in data[(b, c)])),
                                 ood_macro_dsc=float(np.mean([gated(data[(b, c)], .5).mean() for c in ood])),
                                 test_dsc=float(gated(data[(b, test)], .5).mean()))
        entry['detail_test_auc'] = auc(data[(d, test)])
        for mode, theta in thetas.items():
            diffs = {c: gated(data[(d, c)], theta)-gated(data[(b, c)], .5) for c in ood}
            for c in ood:
                if [r['name'] for r in data[(d, c)]] != [r['name'] for r in data[(b, c)]]:
                    raise RuntimeError('Case lists differ between arms')
                averaged[mode][c].append(diffs[c])
            entry[mode] = dict(theta=theta, val_specificity=specificity(data[(d, 'Val')], theta),
                               test_specificity=specificity(data[(d, test)], theta), test_sensitivity=sensitivity(data[(d, test)], theta),
                               test_dsc=float(gated(data[(d, test)], theta).mean()),
                               ood_empty=int(sum(r['score'] <= theta for c in ood for r in data[(d, c)])),
                               ood_macro_dsc=float(np.mean([gated(data[(d, c)], theta).mean() for c in ood])),
                               ood_macro_gain=macro_stats(diffs, rng, draws, perms),
                               ood_pooled_gain=stats(np.concatenate(list(diffs.values())), rng, draws, perms))
        per_seed.append(entry)
    summary = {mode: dict(ood_macro_gain=macro_stats({c: np.mean(v, 0) for c, v in sets.items()}, rng, draws, perms),
                          per_seed_pp=[e[mode]['ood_macro_gain']['mean_pp'] for e in per_seed])
               for mode, sets in averaged.items()}
    plain = [e['ungated']['ood_macro_gain']['mean_pp'] for e in per_seed]
    kept = [e['val_gate']['ood_macro_gain']['mean_pp'] for e in per_seed]
    retained = all(k > 0 for k in kept) and sum(k >= .5*p for k, p in zip(kept, plain)) >= 2
    return dict(per_seed=per_seed, seed_averaged=summary, verdict='retained' if retained else 'reduced')


# ---------------------------------------------------------------- report
def fmt(v, digits=2):
    return 'n/a' if v is None else f'{v:+.{digits}f}'


def ci(s):
    return f"{s['mean_pp']:+.2f} [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}], p={s['p']:.4f}"


def write_report():
    protocol = load_protocol()
    statistics = protocol['statistics']
    rng = np.random.default_rng(statistics['rng_seed'])
    draws, perms = statistics['bootstrap_draws'], statistics['permutation_draws']
    polyps = analyse_polyps(protocol, rng, draws, perms)
    ultra = analyse_ultrasound(protocol, rng, draws, perms)
    timing = {run: read(OUTPUT/'timing'/f'{run}.json') for run in protocol['timing']['runs']}
    plan = read(OUTPUT/'evaluation_plan.json')
    result = dict(name=NAME, created_at=now(), plan_digest=plan['digest'], polyps=polyps, ultrasound=ultra, timing=timing)
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', result)

    lines = [f'# {NAME}: evaluation-only review analyses', '',
             'Nothing was trained. The locked configuration t15 remains the method whatever these numbers show.', '',
             '## A. Polyps: second-pass variants (detail-branch model)', '',
             'Gain = two-pass minus single-pass DSC, percentage points; OOD = equal-weight mean of the four OOD sets.', '',
             '| variant | group | seeds | OOD gain [95% CI], p | per seed | in-domain gain | second pass (OOD) |',
             '|---|---|---|---|---|---|---|']
    for name, block in polyps['variants'].items():
        s = block['all_seeds']
        lines.append(f"| {name} | {block['group']} | {len(s['seeds'])} | {ci(s['ood_macro'])} | "
                     f"{', '.join(fmt(v) for v in s['ood_macro_per_seed_pp'])} | {fmt(s['sets']['Kvasir']['mean_pp'])} | "
                     f"{100*s['second_pass_share_ood']:.0f}% |")
    lines += ['', f"Reference gain on the first three seeds only (for comparison with the sensitivity rows): "
                  f"{polyps['reference_first_three_gain_pp']:+.2f} pp.", '',
              '**Scale or original pixels?**', '']
    for name in ('in224', 'in224_out224'):
        r = polyps['resolution'][name]
        lines.append(f"- {name}: gain {r['gain_pp']:+.2f} pp against {r['reference_gain_pp']:+.2f} pp for t15 "
                     f"(ratio {r['ratio']:.2f}); t15 minus {name}: {ci(r['t15_minus_variant'])}")
    lines += ['', f"Verdict by the rule fixed in advance (ratio of in224): **{polyps['resolution']['verdict']}**.", '',
              f"**Sensitivity:** robust by the rule fixed in advance: **{polyps['sensitivity_robust']}**.", '',
              '## B. Breast ultrasound: detail branch at matched image-level specificity', '',
              'theta = image-score threshold of the detail-branch model; the baseline always uses 0.5.', '',
              '| seed | gate | theta | Val spec. | Test spec. | Test sens. | OOD empty | OOD macro DSC | OOD macro gain [95% CI], p |',
              '|---|---|---|---|---|---|---|---|---|']
    pct = lambda v: 'n/a' if v is None else f'{100*v:.1f}%'
    for e in ultra['per_seed']:
        b = e['baseline']
        lines.append(f"| {e['seed']} | baseline | 0.500 | {pct(b['val_specificity'])} | {pct(b['test_specificity'])} | "
                     f"{pct(b['test_sensitivity'])} | {b['ood_empty']} | {b['ood_macro_dsc']:.2f} | |")
        for mode in ('ungated', 'val_gate', 'test_gate'):
            m = e[mode]
            lines.append(f"| {e['seed']} | detail, {mode} | {m['theta']:.3f} | {pct(m['val_specificity'])} | {pct(m['test_specificity'])} | "
                         f"{pct(m['test_sensitivity'])} | {m['ood_empty']} | {m['ood_macro_dsc']:.2f} | {ci(m['ood_macro_gain'])} |")
    lines += ['', f"Lesion-free images: {ultra['per_seed'][0]['n_val_lesion_free']} in the validation split, "
                  f"{ultra['per_seed'][0]['n_test_lesion_free']} in the test set.", '',
              'Seed-averaged OOD macro gain of the detail branch:', '']
    for mode, s in ultra['seed_averaged'].items():
        lines.append(f"- {mode}: {ci(s['ood_macro_gain'])}; per seed {', '.join(fmt(v) for v in s['per_seed_pp'])}")
    lines += ['', 'Image-level AUC on the BUSI test set (lesion against lesion-free): '
              + '; '.join(f"seed {e['seed']}: baseline {e['baseline']['test_auc']:.3f}, detail {e['detail_test_auc']:.3f}"
                          if e['baseline']['test_auc'] is not None else f"seed {e['seed']}: n/a" for e in ultra['per_seed'])+'.', '',
              f"Verdict by the rule fixed in advance (validation-calibrated gate): **{ultra['verdict']}**.", '',
              '## C. Inference time (batch size 1, 30 stochastic passes per run)', '',
              '| model | device | first pass (ms) | second pass when run (ms) | second pass share | two-pass total (ms) | relative cost |',
              '|---|---|---|---|---|---|---|']
    for run, t in timing.items():
        second = t['second_pass_when_run']
        lines.append(f"| {run} | {t['device']} | {t['first_pass']['mean_ms']:.0f} +/- {t['first_pass']['sd_ms']:.0f} | "
                     + (f"{second['mean_ms']:.0f} +/- {second['sd_ms']:.0f}" if second else 'n/a')
                     + f" | {100*t['second_pass_share']:.0f}% | {t['two_pass_total']['mean_ms']:.0f} | {t['relative_cost_two_pass']:.2f}x |")
    lines += ['', f"Timing images: {protocol['timing']['measured_images']} Kvasir-SEG test images after "
                  f"{protocol['timing']['warmup_images']} warm-up images.", '',
              '## Disclosures', '']+[f'- {d}' for d in protocol['disclosures']]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return str(REPORT/'report.md')
