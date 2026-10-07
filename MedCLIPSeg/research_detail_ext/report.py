"""Tables, paired contrasts and the pre-specified gates for one extension suite. Reads saved cases only."""
import csv
import math

import numpy as np

from research_detail_test.stats import load_cases, arm_summary, contrast, mean_sd
from .common import ROOT, load_suite, output, report_dir, read, maybe, write_json, now

KEYS = [('DSC_percent', 'DSC'), ('small_DSC_percent', 'Small DSC'), ('positive_DSC_percent', 'Positive DSC'),
        ('NSD_percent', 'NSD'), ('micro_foreground_precision_percent', 'Precision'), ('micro_foreground_recall_percent', 'Recall')]


def seed_t(values):
    values = [v for v in values if v is not None]
    if len(values) < 3:
        return None
    sd = float(np.std(values, ddof=1))
    if sd == 0:
        return None
    t = float(np.mean(values)/(sd/math.sqrt(len(values))))
    try:
        from scipy import stats
        return dict(t=t, df=len(values)-1, p=float(2*stats.t.sf(abs(t), len(values)-1)))
    except Exception:
        return dict(t=t, df=len(values)-1, p=None)


def summary(rows, threshold):
    result = arm_summary(rows, threshold)
    positive = [r['DSC'] for r in rows if r['gt_pixels'] > 0]
    result['positive_DSC_percent'] = 100*float(np.mean(positive)) if positive else None
    return result


def analyse(cases, modes, seeds, suite, threshold, groups=None):
    """cases: {mode: {seed: rows}} for one cohort."""
    settings = suite['statistics']
    arms = {}
    for mode in modes:
        per_seed = {seed: summary(cases[mode][seed], threshold) for seed in seeds}
        arms[mode] = {}
        for key in per_seed[seeds[0]]:
            mean, sd = mean_sd([per_seed[s][key] for s in seeds])
            arms[mode][key] = dict(mean=mean, sample_sd=sd, per_seed=[per_seed[s][key] for s in seeds])
    contrasts = {}
    for lhs, rhs in suite['contrasts']:
        if lhs not in modes or rhs not in modes:
            continue
        result = contrast({s: cases[lhs][s] for s in seeds}, {s: cases[rhs][s] for s in seeds}, threshold,
                          settings['permutation_draws'], settings['bootstrap_draws'], settings['rng_seed'], groups)
        result['DSC']['all']['seed_level_t'] = seed_t(result['DSC']['all']['per_seed_mean_pp'])
        positive = [100*(np.mean([r['DSC'] for r in cases[lhs][s] if r['gt_pixels'] > 0])
                         - np.mean([r['DSC'] for r in cases[rhs][s] if r['gt_pixels'] > 0])) for s in seeds]
        result['positive_only_DSC_per_seed_pp'] = [float(v) for v in positive]
        result['negative_false_positive_delta_per_seed'] = [
            arm_summary(cases[lhs][s], threshold)['negative_false_positive']-arm_summary(cases[rhs][s], threshold)['negative_false_positive']
            for s in seeds]
        contrasts[f'{lhs} - {rhs}'] = result
    return dict(arms=arms, contrasts=contrasts, seeds=list(seeds))


def f(value, digits=2, sign=False):
    if value is None:
        return 'n/a'
    return f'{value:+.{digits}f}' if sign else f'{value:.{digits}f}'


def p(value):
    if value is None:
        return 'n/a'
    return '<0.001' if value < .001 else f'{value:.3f}'


def arm_table(result, modes):
    lines = ['| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |',
             '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for mode in modes:
        a = result['arms'][mode]
        cells = [f"{f(a[k]['mean'])} ± {f(a[k]['sample_sd'])}" if a[k]['mean'] is not None else 'n/a' for k, _ in KEYS]
        lines.append(f"| {mode} | "+' | '.join(cells) +
                     f" | {'/'.join(str(v) for v in a['positive_DSC_below_50_percent']['per_seed'])}"
                     f" | {'/'.join(str(v) for v in a['negative_false_positive']['per_seed'])} of {int(a['n_negative']['per_seed'][0])} |")
    return lines


def contrast_table(result, subset, metric='DSC'):
    lines = ['| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |',
             '|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|']
    for name, c in result['contrasts'].items():
        d = c[metric][subset]
        if not d.get('n'):
            lines.append(f'| {name} | 0 |'+' n/a |'*9)
            continue
        lines.append(f"| {name} | {d['n']} | {f(d['mean_pp'], 2, True)} | [{f(d['bootstrap_ci95_pp'][0], 2, True)}, {f(d['bootstrap_ci95_pp'][1], 2, True)}]"
                     f" | {' / '.join(f(v, 2, True) for v in d['per_seed_mean_pp'])} | {d['seeds_positive']}/{len(d['per_seed_mean_pp'])}"
                     f" | {d['improved']}/{d['worsened']}/{d['tied']} | {f(d['median_pp'], 2, True)} | {p(d['permutation_p'])} | {p(d['wilcoxon_p'])}"
                     f" | {f(d['mean_of_others_without_top2_gains_pp'], 2, True)} |")
    return lines


def extra_table(result):
    lines = ['| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |',
             '|---|---|---|---|---|---|']
    for name, c in result['contrasts'].items():
        x = c['pixels_and_failures_per_seed']
        row = lambda key: ' / '.join(f(v, 2, True) for v in x[key])
        lines.append(f"| {name} | {' / '.join(f(v, 2, True) for v in c['positive_only_DSC_per_seed_pp'])} | {row('micro_foreground_precision_percent_delta')}"
                     f" | {row('micro_foreground_recall_percent_delta')} | {' / '.join(f'{v:+d}' for v in c['negative_false_positive_delta_per_seed'])}"
                     f" | {' / '.join(f'{v:+d}' for v in x['positive_DSC_below_50_percent_delta'])} |")
    return lines


def section(title, result, modes):
    return [f'### {title}', '', f"Seeds {', '.join(str(s) for s in result['seeds'])}; mean ± sample SD over seeds.", '',
            *arm_table(result, modes), '', '**Paired DSC differences, all images** (seed-averaged per image).', '',
            *contrast_table(result, 'all'), '', '**Paired DSC differences, small targets only.**', '',
            *contrast_table(result, 'small'), '', '**Other changes.**', '', *extra_table(result), '']


def evaluate_gate(name, gate, results, suite):
    """results: {target: analyse(...)} computed on the seeds the gate refers to."""
    key = ' - '.join(gate['contrast'])
    targets = suite['test_targets']
    dsc = {t: results[t]['contrasts'][key]['DSC']['all']['mean_pp'] for t in targets}
    small = {t: results[t]['contrasts'][key]['DSC']['small'].get('mean_pp') for t in targets}
    per_seed = np.array([results[t]['contrasts'][key]['DSC']['all']['per_seed_mean_pp'] for t in targets])
    macro_per_seed = per_seed.mean(0)
    checks = {}
    if 'source_test_dsc_gain_gt_pp' in gate:
        value = dsc[targets[0]]
        checks['source_test_dsc_gain'] = dict(value=value, passed=value > gate['source_test_dsc_gain_gt_pp'])
    if 'min_ood_sets_positive' in gate:
        positive = [t for t in suite['ood_targets'] if dsc[t] > 0]
        checks['ood_sets_positive'] = dict(value=len(positive), sets=positive, passed=len(positive) >= gate['min_ood_sets_positive'])
    if 'min_sets_positive' in gate:
        positive = [t for t in targets if dsc[t] > 0]
        checks['sets_positive'] = dict(value=len(positive), sets=positive, passed=len(positive) >= gate['min_sets_positive'])
    if 'macro_dsc_gain_gt_pp' in gate:
        value = float(np.mean(list(dsc.values())))
        checks['macro_dsc_gain'] = dict(value=value, passed=value > gate['macro_dsc_gain_gt_pp'])
    if 'macro_small_dsc_gain_gt_pp' in gate:
        values = [v for v in small.values() if v is not None]
        value = float(np.mean(values))
        checks['macro_small_dsc_gain'] = dict(value=value, passed=value > gate['macro_small_dsc_gain_gt_pp'])
    if 'guard_max_regression_pp' in gate:
        value = min(dsc.values())
        checks['guard_no_large_regression'] = dict(value=value, passed=value >= -gate['guard_max_regression_pp'])
    if 'min_seeds_positive_macro' in gate:
        value = int((macro_per_seed > 0).sum())
        checks['seeds_positive_macro'] = dict(value=value, of=len(macro_per_seed), passed=value >= gate['min_seeds_positive_macro'])
    return dict(name=name, contrast=key, dsc_gain_pp=dsc, small_dsc_gain_pp=small,
                macro_per_seed_pp=[float(v) for v in macro_per_seed], macro_seed_level_t=seed_t(list(macro_per_seed)),
                checks=checks, passed=all(c['passed'] for c in checks.values()))


def earlier_cases(mode, seed, cohort):
    """Seeds 666-668 of formal_detail_mix_kvasir_v1: accepted Val cases and the accepted Test/OOD evaluation."""
    from research_detail_test import common as earlier
    run = f'{mode}_seed{seed}'
    if cohort == 'Val':
        return load_cases(earlier.val_cases_path(run))
    folder = earlier.OUTPUT/'evaluations'/run/cohort
    if not maybe(folder/'manifest.json').get('complete'):
        raise ValueError(f'Earlier evaluation missing: {run}/{cohort}')
    return load_cases(folder/'cases.csv')


def write_report(name):
    suite = load_suite(name)
    plan = read(output(suite)/'experiment_plan.json')
    threshold, modes, new = plan['small_threshold'], suite['modes'], list(suite['seeds'])
    old = list(suite.get('combine_with_earlier_seeds', []))
    cohorts = ['Val']+suite['test_targets']

    def cases_for(cohort, seeds):
        table = {mode: {} for mode in modes}
        for mode in modes:
            for seed in seeds:
                if seed in new:
                    folder = output(suite)/'evaluations'/f'{mode}_seed{seed}'/cohort
                    if not maybe(folder/'manifest.json').get('complete'):
                        raise ValueError(f'Incomplete evaluation: {mode}_seed{seed}/{cohort}')
                    table[mode][seed] = load_cases(folder/'cases.csv')
                else:
                    table[mode][seed] = earlier_cases(mode, seed, cohort)
        return table

    views = {'new': new}
    if old:
        views['all'] = old+new
    results, rows = {}, []
    for view, seeds in views.items():
        results[view] = {}
        for cohort in cohorts:
            cases = cases_for(cohort, seeds)
            groups = list(plan['test_cohorts'][cohort]['image_hash_groups'].values()) if cohort != 'Val' else None
            results[view][cohort] = analyse(cases, modes, seeds, suite, threshold, groups)
            if view == 'new':
                for mode in modes:
                    for seed in seeds:
                        rows += [dict(dataset=cohort, arm=mode, seed=seed, **r) for r in cases[mode][seed]]
    gates = {}
    for gate_name, gate in suite['gates'].items():
        view = 'all' if gate.get('seeds') == 'all' else 'new'
        gates[gate_name] = dict(evaluate_gate(gate_name, gate, results[view], suite), seeds=views[view])
    destination = report_dir(suite)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'results.json', dict(suite=suite, plan_digest=plan['digest'], small_threshold=threshold, gates=gates,
                                                results=results, overlap_with_source={t: plan['test_cohorts'][t]['overlap_with_source_train_val'] for t in suite['test_targets']},
                                                generated_at=now()))
    with (destination/'all_cases.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mark = lambda ok: 'PASS' if ok else 'FAIL'
    lines = [f"# {suite['name']}: results", '',
             f"Source {suite['dataset']}; arms {', '.join(modes)}; new seeds {', '.join(str(s) for s in new)}; epoch 100; MC30, batch 32, threshold 0.5; fixed text `{suite['prompt']}`.",
             f"Small target: foreground fraction <= {threshold:.6f} (source Train positive 25th percentile). Gates and the final-model rule were fixed before training.", '',
             '## Gates', '']
    for gate_name, gate in gates.items():
        lines += [f"**{gate_name}** ({gate['contrast']}, seeds {', '.join(str(s) for s in gate['seeds'])}): **{mark(gate['passed'])}**", '',
                  '| Criterion | Value | Result |', '|---|---|---|']
        for check, c in gate['checks'].items():
            value = f"{c['value']} ({', '.join(c['sets']) or 'none'})" if 'sets' in c else (f"{c['value']} of {c['of']}" if 'of' in c else f"{f(c['value'], 2, True)} pp")
            lines.append(f"| {check} | {value} | {mark(c['passed'])} |")
        t = gate['macro_seed_level_t']
        lines += ['', f"Macro gain per seed: {' / '.join(f(v, 2, True) for v in gate['macro_per_seed_pp'])} pp; seed-level paired t p = {p(t['p']) if t else 'n/a'}.", '',
                  '| Dataset | DSC gain pp | Small-target DSC gain pp |', '|---|---:|---:|']
        lines += [f"| {t_} | {f(gate['dsc_gain_pp'][t_], 2, True)} | {f(gate['small_dsc_gain_pp'][t_], 2, True)} |" for t_ in suite['test_targets']]
        lines.append('')
    lines += [f"Final-model rule: {suite['final_model_rule']}", '']
    overlaps = {t: len(plan['test_cohorts'][t]['overlap_with_source_train_val']) for t in suite['test_targets']}
    shared = {t: plan['test_cohorts'][t]['shared_unique_images_with_other_tests'] for t in suite['test_targets']}
    lines += ['## Cohort notes', '', '| Dataset | Records | Unique images | Exact overlap with source Train/Val | Shared with other test sets |', '|---|---:|---:|---:|---|']
    lines += [f"| {t} | {plan['test_cohorts'][t]['n']} | {plan['test_cohorts'][t]['n_unique_images']} | {overlaps[t]} | {shared[t] or 'none'} |" for t in suite['test_targets']]
    lines.append('')
    for view, seeds in views.items():
        lines += [f"## {'New seeds only' if view == 'new' else 'All seeds combined'}", '']
        for cohort in cohorts:
            label = 'source Val (development)' if cohort == 'Val' else ('Test, source domain' if cohort == suite['dataset'] else 'Test, OOD')
            lines += section(f'{cohort} ({label})', results[view][cohort], modes)
    lines += ['## Limitations', '', '- Image-level tests are conditional on the trained seeds, not patient clustered, and not corrected for multiple comparisons.',
              *[f'- {text}' for text in suite.get('disclosures', [])], '']
    (destination/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return destination/'report.md'
