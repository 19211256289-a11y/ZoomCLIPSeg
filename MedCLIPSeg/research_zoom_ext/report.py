"""Paired two-pass versus single-pass statistics per arm, and the pre-specified gate where a suite has one. numpy only."""
import csv

import numpy as np

from research_detail_test.stats import FLOATS, INTS, contrast
from .common import Source, load_suite, output, report_dir, split_run, read, maybe, write_json, now

BINS = ((0, .01, '<1%'), (.01, .03, '1-3%'), (.03, .06, '3-6%'), (.06, .15, '6-15%'), (.15, 1.01, '>15%'))


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


def analyse(cases, suite, targets, small, groups, configs):
    """cases: {(run, target): rows}. Returns results for every config and arm, and the gate if the suite has one."""
    stat, seeds, ood = suite['statistics'], suite['seeds'], targets[1:]
    mean = lambda values: float(np.mean(values)) if len(values) else None
    arms = {}
    for arm in suite['arms']:
        pick = lambda target, name: {s: [r for r in cases[f'{arm}_seed{s}', target] if r['config'] == name] for s in seeds}
        arms[arm] = {}
        for config in configs:
            per_target = {}
            for target in targets:
                two, one = pick(target, config), pick(target, 'single')
                result = contrast(two, one, small, stat['permutation_draws'], stat['bootstrap_draws'], stat['rng_seed'],
                                  groups=groups.get(target))
                over = lambda arm_rows, keep: mean([100*np.mean([r['DSC'] for r in arm_rows[s] if keep(r)]) for s in seeds
                                                    if any(keep(r) for r in arm_rows[s])])
                everything, positive = (lambda r: True), (lambda r: r['gt_pixels'] > 0)
                is_small = lambda r: r['gt_pixels'] > 0 and r['gt_fraction'] <= small
                negative = lambda r: r['gt_pixels'] == 0
                count = lambda arm_rows, test: mean([sum(test(r) for r in arm_rows[s]) for s in seeds])
                per_target[target] = dict(
                    n=len(one[seeds[0]]), n_positive=sum(positive(r) for r in one[seeds[0]]),
                    n_small=sum(is_small(r) for r in one[seeds[0]]),
                    single_DSC=over(one, everything), two_pass_DSC=over(two, everything),
                    single_NSD=mean([100*np.mean([r['NSD'] for r in one[s]]) for s in seeds]),
                    two_pass_NSD=mean([100*np.mean([r['NSD'] for r in two[s]]) for s in seeds]),
                    single_positive_DSC=over(one, positive), two_pass_positive_DSC=over(two, positive),
                    single_small_DSC=over(one, is_small), two_pass_small_DSC=over(two, is_small),
                    failures_single=count(one, lambda r: r['gt_pixels'] > 0 and r['DSC'] < .5),
                    failures_two_pass=count(two, lambda r: r['gt_pixels'] > 0 and r['DSC'] < .5),
                    normal_false_positive_single=count(one, lambda r: negative(r) and r['pred_pixels'] > 0),
                    normal_false_positive_two_pass=count(two, lambda r: negative(r) and r['pred_pixels'] > 0),
                    zoomed_share=mean([np.mean([r['kind'] == 'in' for r in two[s]]) for s in seeds]),
                    second_pass_empty=count(two, lambda r: r['fallback']), contrast=result)
            gain = {t: per_target[t]['contrast']['DSC']['all']['mean_pp'] for t in targets}
            per_seed = np.mean([per_target[t]['contrast']['DSC']['all']['per_seed_mean_pp'] for t in ood], axis=0)
            strata = []
            for low, high, label in BINS:
                one, two = [], []
                for s in seeds:
                    for target in ood:
                        rows = cases[f'{arm}_seed{s}', target]
                        keep = {r['name'] for r in rows if r['config'] == 'single' and r['gt_pixels'] > 0 and low <= r['gt_fraction'] < high}
                        one += [r for r in rows if r['config'] == 'single' and r['name'] in keep]
                        two += [r for r in rows if r['config'] == config and r['name'] in keep]
                if one:
                    strata.append(dict(stratum=label, n=len(one)/len(seeds), single_DSC=100*mean([r['DSC'] for r in one]),
                                       two_pass_DSC=100*mean([r['DSC'] for r in two]),
                                       zoomed_share=mean([r['kind'] == 'in' for r in two]),
                                       failures_single=sum(r['DSC'] < .5 for r in one)/len(seeds),
                                       failures_two_pass=sum(r['DSC'] < .5 for r in two)/len(seeds)))
            arms[arm][config] = dict(targets=per_target, gain_pp=gain, macro_ood_gain_pp=mean([gain[t] for t in ood]),
                                     macro_ood_gain_per_seed_pp=[float(v) for v in per_seed], ood_lesion_size_strata=strata)
    result = dict(arms=arms, seeds=seeds, targets=targets, primary_config=suite['primary_config'], verdict='descriptive')
    gate = suite['gate']
    if gate:
        chosen = arms[gate['primary_arm']][suite['primary_config']]
        gain = chosen['gain_pp']
        checks = dict(
            g1=bool(chosen['macro_ood_gain_pp'] >= gate['g1_macro_ood_dsc_gain_at_least_pp']),
            g2=bool(sum(gain[t] > 0 for t in ood) >= gate['g2_min_ood_sets_with_positive_gain']),
            g3=bool(gain[targets[0]] >= gate['g3_source_test_dsc_change_at_least_pp']),
            g4=bool(min(gain.values()) >= -gate['g4_max_dsc_regression_any_set_pp']),
            g5=bool(all(v > 0 for v in chosen['macro_ood_gain_per_seed_pp'])))
        result.update(checks=checks, verdict='pass' if all(checks.values()) else 'fail', primary_arm=gate['primary_arm'])
    return result


def write_report(name):
    suite = load_suite(name)
    source = Source(suite)
    plan = read(output(name)/'evaluation_plan.json')
    cases = {}
    for run in suite['runs']:
        for target in source.targets:
            folder = output(name)/'evaluations'/run/target
            if not maybe(folder/'manifest.json').get('complete') or not maybe(folder/'reproduction.json').get('passed'):
                raise ValueError('Evaluation not complete or single-pass check not passed: '+str(folder))
            cases[run, target] = load(folder/'cases.csv')
    cohorts = source.plan()['test_cohorts']
    groups = {t: list(cohorts[t]['image_hash_groups'].values()) for t in source.targets}
    configs = [c['name'] for c in plan['configs']]
    result = analyse(cases, suite, source.targets, plan['small_threshold'], groups, configs)
    result.update(name=name, created_at=now(), plan_digest=plan['digest'], configs=plan['configs'], calibration=plan['calibration'],
                  reproduction={f'{run}/{target}': read(output(name)/'evaluations'/run/target/'reproduction.json')['difference_pp']
                                for run in suite['runs'] for target in source.targets})
    destination = report_dir(name)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'results.json', result)
    targets, ood, primary = source.targets, source.ood, suite['primary_config']
    fmt = lambda v, s='.2f': 'n/a' if v is None else format(v, s)
    seeds = ', '.join(str(s) for s in suite['seeds'])
    lines = [f'# {name}: locked two-pass zoom, evaluation only', '', suite['purpose'].strip(), '',
             f'Seeds {seeds}; means over seeds; contrast is two-pass minus single-pass on the same checkpoints.',
             f"Fixed sentence: \"{plan['prompt']}\". Configs: "
             + '; '.join(f"{c['name']} (target fraction {c['target_fraction']:.4f})" for c in plan['configs'])
             + f'. Primary config: {primary}.', '']
    if plan['calibration']:
        c = plan['calibration']
        lines += [f"Source calibration from Train masks only: 0.15 is the {100*c['reference_quantile_rank']:.1f}th percentile of Kvasir Train "
                  f"lesion fractions; the same percentile of the source Train set is {c['target_fraction']:.4f} "
                  f"(medians: Kvasir {c['reference_median']:.4f}, source {c['source_median']:.4f}).", '']
    if suite['gate']:
        chosen = result['arms'][result['primary_arm']][primary]
        lines += [f"**Verdict: {result['verdict']}** (arm {result['primary_arm']}, config {primary}): macro OOD DSC gain "
                  f"{chosen['macro_ood_gain_pp']:+.2f} pp (per seed {', '.join(f'{v:+.2f}' for v in chosen['macro_ood_gain_per_seed_pp'])})", '']
    else:
        lines += ['**No gate: descriptive ablation.**', '']
    for config in configs:
        lines += [f'## DSC by arm, config {config}' + (' (primary)' if config == primary else ' (secondary)'), '',
                  '| arm | ' + ' | '.join(f'{t} single | {t} two-pass | gain' for t in targets) + ' | macro OOD gain | per seed |',
                  '|---|' + '---|---|---|'*len(targets) + '---|---|']
        for arm in suite['arms']:
            r = result['arms'][arm][config]
            cells = ' | '.join(f"{r['targets'][t]['single_DSC']:.2f} | {r['targets'][t]['two_pass_DSC']:.2f} | {r['gain_pp'][t]:+.2f}" for t in targets)
            lines.append(f"| {arm} | {cells} | {r['macro_ood_gain_pp']:+.2f} | {', '.join(f'{v:+.2f}' for v in r['macro_ood_gain_per_seed_pp'])} |")
        lines.append('')
    for arm in suite['arms']:
        r = result['arms'][arm][primary]
        lines += [f'## Arm {arm}, config {primary}: details', '',
                  '| set | n | gain (pp) | per seed | 95% CI | perm. p | better / worse | NSD gain | lesion-only gain | small n | small gain | failures | normal-image FP | second pass on |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for t in targets:
            x, d = r['targets'][t], r['targets'][t]['contrast']['DSC']['all']
            delta = lambda a, b: 'n/a' if x[a] is None or x[b] is None else f'{x[b]-x[a]:+.2f}'
            lines.append(f"| {t} | {x['n']} | {d['mean_pp']:+.2f} | {', '.join(f'{v:+.2f}' for v in d['per_seed_mean_pp'])} | "
                         f"{d['bootstrap_ci95_pp'][0]:+.2f} to {d['bootstrap_ci95_pp'][1]:+.2f} | {d['permutation_p']:.4f} | "
                         f"{d['improved']} / {d['worsened']} | {delta('single_NSD', 'two_pass_NSD')} | "
                         f"{delta('single_positive_DSC', 'two_pass_positive_DSC')} | {x['n_small']} | "
                         f"{delta('single_small_DSC', 'two_pass_small_DSC')} | {x['failures_single']:.1f} -> {x['failures_two_pass']:.1f} | "
                         f"{x['normal_false_positive_single']:.1f} -> {x['normal_false_positive_two_pass']:.1f} | {100*x['zoomed_share']:.0f}% |")
        lines += ['', '| OOD lesion share of image | n | single DSC | two-pass DSC | zoomed | failures single -> two-pass |', '|---|---|---|---|---|---|']
        for s in r['ood_lesion_size_strata']:
            lines.append(f"| {s['stratum']} | {s['n']:.0f} | {s['single_DSC']:.2f} | {s['two_pass_DSC']:.2f} | {100*s['zoomed_share']:.0f}% | "
                         f"{s['failures_single']:.1f} -> {s['failures_two_pass']:.1f} |")
        lines.append('')
    if suite['gate']:
        gate, c = suite['gate'], result['checks']
        chosen = result['arms'][result['primary_arm']][primary]
        lines += ['## Gate (fixed in advance)', '',
                  f"- g1 macro OOD gain {chosen['macro_ood_gain_pp']:+.2f} pp (need >= {gate['g1_macro_ood_dsc_gain_at_least_pp']:+.1f}): {c['g1']}",
                  f"- g2 OOD sets with a positive gain {sum(chosen['gain_pp'][t] > 0 for t in ood)}/{len(ood)} (need >= {gate['g2_min_ood_sets_with_positive_gain']}): {c['g2']}",
                  f"- g3 source Test change {chosen['gain_pp'][targets[0]]:+.2f} pp (need >= {gate['g3_source_test_dsc_change_at_least_pp']:+.1f}): {c['g3']}",
                  f"- g4 largest regression on any set {min(chosen['gain_pp'].values()):+.2f} pp (need >= {-gate['g4_max_dsc_regression_any_set_pp']:+.1f}): {c['g4']}",
                  f"- g5 every seed positive on the macro OOD gain: {c['g5']}", '']
    lines += [f"Single-pass results reproduce the accepted evaluation within {max(abs(v) for v in result['reproduction'].values()):.4f} pp.",
              'Disclosures: ' + ' '.join(suite['disclosures']), '']
    (destination/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return destination/'report.md'
