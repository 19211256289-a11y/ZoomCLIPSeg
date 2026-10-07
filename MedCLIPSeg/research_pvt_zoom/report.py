"""Report of SCZ on Polyp-PVT from saved per-image files. Rows of our models are accepted results of the same seeds."""
import csv

import numpy as np

from research_polyp_pvt import common as pvt
from research_polyp_pvt.report import stats, macro_stats
from .common import *


def load(path, config=None):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if config is not None:
        rows = [r for r in rows if r['config'] == config]
    return {r['name']: dict(DSC=100*float(r['DSC']), gt_fraction=float(r['gt_fraction']), gt_pixels=int(r['gt_pixels']),
                            zoomed=r.get('kind') == 'in') for r in rows}


def analyse(protocol, source):
    statistics, seeds = protocol['statistics'], protocol['seeds']
    rng = np.random.default_rng(statistics['rng_seed'])
    draws, perms = statistics['bootstrap_draws'], statistics['permutation_draws']
    models = {'pvt': lambda s, t: load(evaluation_dir(s, t)/'cases.csv', 'single'),
              'pvt_scz': lambda s, t: load(evaluation_dir(s, t)/'cases.csv', 't15'),
              'ours_baseline': lambda s, t: load(*pvt.accepted_cases(source, 'baseline', s, t)),
              'ours_full': lambda s, t: load(*pvt.accepted_cases(source, 'full', s, t))}
    cases = {m: {t: {s: f(s, t) for s in seeds} for t in ALL} for m, f in models.items()}
    names = {t: sorted(cases['pvt'][t][seeds[0]]) for t in ALL}
    for m in cases:
        for t in ALL:
            for s in seeds:
                if sorted(cases[m][t][s]) != names[t]:
                    raise RuntimeError(f'Case list differs: {m}, {t}, seed {s}')
    mean = lambda m, t, keep=None: float(np.mean([cases[m][t][s][n]['DSC'] for s in seeds for n in (keep or names[t])]))
    summary = {m: dict(sets={t: mean(m, t) for t in ALL}, ood=float(np.mean([mean(m, t) for t in OOD]))) for m in cases}
    diff = {t: np.array([np.mean([cases['pvt_scz'][t][s][n]['DSC']-cases['pvt'][t][s][n]['DSC'] for s in seeds]) for n in names[t]]) for t in ALL}
    per_seed = [float(np.mean([np.mean([cases['pvt_scz'][t][s][n]['DSC']-cases['pvt'][t][s][n]['DSC'] for n in names[t]]) for t in OOD])) for s in seeds]
    gain = dict(sets={t: stats(diff[t], rng, draws, perms) for t in ALL}, ood_macro=macro_stats({t: diff[t] for t in OOD}, rng, draws, perms),
                ood_macro_per_seed_pp=per_seed,
                second_pass_share={t: float(np.mean([cases['pvt_scz'][t][s][n]['zoomed'] for s in seeds for n in names[t]])) for t in ALL})
    c = gain['ood_macro']
    verdict = ('scz_transfers' if c['mean_pp'] > 0 and c['ci95_pp'][0] > 0 and all(v > 0 for v in per_seed) else
               'scz_harms' if c['mean_pp'] < 0 and c['ci95_pp'][1] < 0 else 'no_effect')
    edges = [0.]+[e/100 for e in protocol['strata_percent']]+[1.01]
    labels = [f'<{protocol["strata_percent"][0]}%']+[f'{a}-{b}%' for a, b in zip(protocol['strata_percent'], protocol['strata_percent'][1:])] \
        + [f'>={protocol["strata_percent"][-1]}%']
    strata = {}
    for scope, targets in (('OOD pooled', OOD), (HELD_OUT, (HELD_OUT,))):
        rows = []
        for label, low, high in zip(labels, edges, edges[1:]):
            keep = {t: [n for n in names[t] if cases['pvt'][t][seeds[0]][n]['gt_pixels'] > 0
                        and low <= cases['pvt'][t][seeds[0]][n]['gt_fraction'] < high] for t in targets}
            n = sum(len(v) for v in keep.values())
            rows.append(dict(stratum=label, n=n, **{m: (float(np.mean([cases[m][t][s][k]['DSC'] for t in targets for s in seeds for k in keep[t]]))
                                                    if n else None) for m in cases}))
        strata[scope] = rows
    return dict(summary=summary, scz_on_pvt=gain, verdict=verdict, strata=strata)


def ci(s):
    return f"{s['mean_pp']:+.2f} [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}], p={s['p']:.4f}"


def write_report():
    protocol, source = load_protocol()
    analysis = analyse(protocol, source)
    plan = read(OUTPUT/'plan.json')
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', dict(name=NAME, created_at=now(), plan_digest=plan['digest'], **analysis))
    labels = dict(pvt='Polyp-PVT, single pass', pvt_scz='Polyp-PVT + SCZ', ours_baseline='MedCLIPSeg baseline, single pass',
                  ours_full='ZoomCLIPSeg (detail branch + SCZ)')
    summary, gain = analysis['summary'], analysis['scz_on_pvt']
    lines = [f'# {NAME}: the locked SCZ applied, unchanged, to Polyp-PVT (three seeds)', '',
             'Nothing was trained. DSC in %, mean over seeds and images; OOD = equal-weight mean of the four OOD sets.', '',
             '| model | '+' | '.join(TARGETS)+f' | OOD mean | {HELD_OUT} |', '|---|'+'---|'*(len(TARGETS)+2)]
    for m, label in labels.items():
        lines.append(f'| {label} | '+' | '.join(f"{summary[m]['sets'][t]:.2f}" for t in TARGETS)
                     + f" | {summary[m]['ood']:.2f} | {summary[m]['sets'][HELD_OUT]:.2f} |")
    lines += ['', '## Gain of SCZ on Polyp-PVT (two-pass minus single-pass, pp) [95% CI], p', '',
              '| | '+' | '.join(ALL)+' | OOD mean |', '|---|'+'---|'*(len(ALL)+1),
              '| gain | '+' | '.join(ci(gain['sets'][t]) for t in ALL)+f" | {ci(gain['ood_macro'])} |",
              '| second pass | '+' | '.join(f"{100*gain['second_pass_share'][t]:.0f}%" for t in ALL)+' | |', '',
              f"OOD gain per seed: {', '.join(f'{v:+.2f}' for v in gain['ood_macro_per_seed_pp'])}", '',
              f"**Verdict by the rule fixed in advance: {analysis['verdict']}.**", '',
              f"Consequence fixed in advance: {protocol['reading_rules_fixed_in_advance']['consequence']}", '',
              '## DSC by lesion area as a share of the image', '']
    for scope, rows in analysis['strata'].items():
        lines += [f'{scope}:', '', '| lesion area | n | '+' | '.join(labels.values())+' |', '|---|---|'+'---|'*len(labels)]
        for row in rows:
            lines.append(f"| {row['stratum']} | {row['n']} | "+' | '.join('' if row[m] is None else f'{row[m]:.2f}' for m in labels)+' |')
        lines.append('')
    lines += ['## Disclosures', '']+[f'- {d}' for d in protocol['disclosures']]+[f"- {protocol['locked_before']}"]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return str(REPORT/'report.md')
