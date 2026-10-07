"""Paired two-pass versus single-pass statistics and the pre-specified gate. numpy only."""
import csv

import numpy as np

from research_detail_test.stats import FLOATS, INTS, contrast
from .common import (OUTPUT, REPORT, RUNS, OOD, TARGETS, SMALL, ACCEPTED_TEST, load_protocol, write_json, now, maybe, read,
                     split_run)

BINS = ((0, .01, '<1%'), (.01, .03, '1-3%'), (.03, SMALL, '3-5.9%'), (SMALL, .15, '5.9-15%'), (.15, 1.01, '>15%'))


def load(path):
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as stream:
        for r in csv.DictReader(stream):
            row = dict(config=r['config'], name=r['name'], image_name=r['image_name'], kind=r['kind'],
                       window=float(r['window']), fallback=int(float(r['fallback'])))
            for k in INTS:
                row[k] = int(float(r[k]))
            for k in FLOATS:
                row[k] = float(r[k]) if r[k] not in ('', 'None') else None
            rows.append(row)
    return rows


def analyse(cases, protocol, groups):
    """cases: {(run, target): rows}; groups: {target: lists of image names with identical content}."""
    config, stat, gate = protocol['config']['name'], protocol['statistics'], protocol['gate']
    seeds = [split_run(run)[1] for run in RUNS]
    pick = lambda target, name: {split_run(run)[1]: [r for r in cases[run, target] if r['config'] == name] for run in RUNS}
    targets, per_seed_gain = {}, {}
    for target in TARGETS:
        two, one = pick(target, config), pick(target, 'single')
        result = contrast(two, one, SMALL, stat['permutation_draws'], stat['bootstrap_draws'], stat['rng_seed'],
                          groups=groups.get(target))
        mean = lambda arm, rows_filter=lambda r: True: float(np.mean(
            [100*np.mean([r['DSC'] for r in arm[s] if rows_filter(r)]) for s in seeds]))
        small = lambda r: r['gt_pixels'] > 0 and r['gt_fraction'] <= SMALL
        failures = lambda arm: float(np.mean([sum(r['gt_pixels'] > 0 and r['DSC'] < .5 for r in arm[s]) for s in seeds]))
        n = len(one[seeds[0]])
        targets[target] = dict(
            n=n, single_DSC=mean(one), two_pass_DSC=mean(two),
            single_NSD=float(np.mean([100*np.mean([r['NSD'] for r in one[s]]) for s in seeds])),
            two_pass_NSD=float(np.mean([100*np.mean([r['NSD'] for r in two[s]]) for s in seeds])),
            n_small=sum(small(r) for r in one[seeds[0]]),
            single_small_DSC=mean(one, small) if any(small(r) for r in one[seeds[0]]) else None,
            two_pass_small_DSC=mean(two, small) if any(small(r) for r in one[seeds[0]]) else None,
            failures_single=failures(one), failures_two_pass=failures(two),
            zoomed_share=float(np.mean([np.mean([r['kind'] == 'in' for r in two[s]]) for s in seeds])),
            second_pass_empty=float(np.mean([sum(r['fallback'] for r in two[s]) for s in seeds])),
            contrast=result)
        per_seed_gain[target] = result['DSC']['all']['per_seed_mean_pp']
    gain = {t: targets[t]['contrast']['DSC']['all']['mean_pp'] for t in TARGETS}
    macro_seed = np.mean([per_seed_gain[t] for t in OOD], axis=0)
    macro = float(np.mean([gain[t] for t in OOD]))
    checks = dict(
        g1=bool(macro >= gate['g1_macro_ood_dsc_gain_at_least_pp']),
        g2=bool(sum(gain[t] > 0 for t in OOD) >= gate['g2_min_ood_sets_with_positive_gain']),
        g3=bool(gain['Kvasir'] >= gate['g3_source_test_dsc_change_at_least_pp']),
        g4=bool(min(gain.values()) >= -gate['g4_max_dsc_regression_any_set_pp']),
        g5=bool((macro_seed > 0).all()))
    strata = []
    for low, high, label in BINS:
        one, two = [], []
        for run in RUNS:
            for target in OOD:
                rows = cases[run, target]
                keep = {r['name'] for r in rows if r['config'] == 'single' and r['gt_pixels'] > 0 and low <= r['gt_fraction'] < high}
                one += [r for r in rows if r['config'] == 'single' and r['name'] in keep]
                two += [r for r in rows if r['config'] == config and r['name'] in keep]
        if one:
            strata.append(dict(stratum=label, n=len(one)/len(RUNS), single_DSC=100*float(np.mean([r['DSC'] for r in one])),
                               two_pass_DSC=100*float(np.mean([r['DSC'] for r in two])),
                               zoomed_share=float(np.mean([r['kind'] == 'in' for r in two])),
                               failures_single=sum(r['DSC'] < .5 for r in one)/len(RUNS),
                               failures_two_pass=sum(r['DSC'] < .5 for r in two)/len(RUNS)))
    return dict(config=config, targets=targets, gain_pp=gain, macro_ood_gain_pp=macro,
                macro_ood_gain_per_seed_pp=[float(v) for v in macro_seed], seeds=seeds, checks=checks,
                verdict='pass' if all(checks.values()) else 'fail', ood_lesion_size_strata=strata)


def write_report():
    protocol = load_protocol()
    cases = {}
    for run in RUNS:
        for target in TARGETS:
            folder = OUTPUT/'evaluations'/run/target
            if not maybe(folder/'manifest.json').get('complete') or not maybe(folder/'reproduction.json').get('passed'):
                raise ValueError('Evaluation not complete or single-pass check not passed: '+str(folder))
            cases[run, target] = load(folder/'cases.csv')
    cohorts = read(ACCEPTED_TEST/'evaluation_plan.json')['test_cohorts']
    groups = {t: list(cohorts[t]['image_hash_groups'].values()) for t in TARGETS}
    result = analyse(cases, protocol, groups)
    result.update(name=protocol['name'], created_at=now(), reproduction={
        f'{run}/{target}': read(OUTPUT/'evaluations'/run/target/'reproduction.json')['difference_pp']
        for run in RUNS for target in TARGETS})
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', result)
    fmt = lambda v, s='.2f': 'n/a' if v is None else format(v, s)
    lines = ['# Two-pass zoom: one-time Test confirmation (detail arm, locked config t15)', '',
             'Evaluation only. Three-seed mean. Contrast: two-pass minus single-pass on the same checkpoints.', '',
             f"**Verdict: {result['verdict']}** | macro OOD DSC gain {result['macro_ood_gain_pp']:+.2f} pp "
             f"(per seed {', '.join(f'{v:+.2f}' for v in result['macro_ood_gain_per_seed_pp'])})", '',
             '## DSC and NSD', '',
             '| set | n | single DSC | two-pass DSC | gain (pp) | per seed | 95% CI | perm. p | NSD gain | better / worse images |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for t in TARGETS:
        r, d = result['targets'][t], result['targets'][t]['contrast']['DSC']['all']
        lines.append(f"| {t} | {r['n']} | {r['single_DSC']:.2f} | {r['two_pass_DSC']:.2f} | {d['mean_pp']:+.2f} | "
                     f"{', '.join(f'{v:+.2f}' for v in d['per_seed_mean_pp'])} | {d['bootstrap_ci95_pp'][0]:+.2f} to {d['bootstrap_ci95_pp'][1]:+.2f} | "
                     f"{d['permutation_p']:.4f} | {r['two_pass_NSD']-r['single_NSD']:+.2f} | {d['improved']} / {d['worsened']} |")
    lines += ['', '## Small targets, failures and cost', '',
              '| set | small n | small DSC single | small DSC two-pass | failures single -> two-pass | share of images with a second pass | second pass empty |',
              '|---|---|---|---|---|---|---|']
    for t in TARGETS:
        r = result['targets'][t]
        lines.append(f"| {t} | {r['n_small']} | {fmt(r['single_small_DSC'])} | {fmt(r['two_pass_small_DSC'])} | "
                     f"{r['failures_single']:.1f} -> {r['failures_two_pass']:.1f} | {100*r['zoomed_share']:.0f}% | {r['second_pass_empty']:.1f} |")
    lines += ['', '## OOD by lesion size (four OOD sets pooled, means over seeds)', '',
              '| lesion share of image | n | single DSC | two-pass DSC | zoomed | failures single -> two-pass |', '|---|---|---|---|---|---|']
    for s in result['ood_lesion_size_strata']:
        lines.append(f"| {s['stratum']} | {s['n']:.0f} | {s['single_DSC']:.2f} | {s['two_pass_DSC']:.2f} | {100*s['zoomed_share']:.0f}% | "
                     f"{s['failures_single']:.1f} -> {s['failures_two_pass']:.1f} |")
    unique = {t: result['targets'][t]['contrast']['DSC'].get('unique_images') for t in ('ColonDB', 'CVC300')}
    c = result['checks']
    lines += ['', 'Unique-image sensitivity (equal weight per distinct image): '
              + '; '.join(f"{t} {u['mean_pp']:+.2f} pp over {u['n']} images" for t, u in unique.items() if u), '',
              '## Gate (fixed before any two-pass Test number was computed)', '',
              f"- g1 macro OOD gain {result['macro_ood_gain_pp']:+.2f} pp (need >= +1.0): {c['g1']}",
              f"- g2 OOD sets with a positive gain {sum(result['gain_pp'][t] > 0 for t in OOD)}/4 (need >= 3): {c['g2']}",
              f"- g3 Kvasir Test change {result['gain_pp']['Kvasir']:+.2f} pp (need >= -0.2): {c['g3']}",
              f"- g4 largest regression on any set {min(result['gain_pp'].values()):+.2f} pp (need >= -0.5): {c['g4']}",
              f"- g5 every seed positive on the macro OOD gain: {c['g5']}", '',
              f"Single-pass results reproduce the accepted Test evaluation within {max(abs(v) for v in result['reproduction'].values()):.4f} pp.",
              'Disclosures: the weakness was found on these Test predictions; the configuration was selected on rescaled source Val',
              'images only; the test sets were used before; ColonDB and CVC300 share 52 images; the second pass costs one extra',
              'MC30 inference per zoomed image.', '']
    (REPORT/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return REPORT/'report.md'
