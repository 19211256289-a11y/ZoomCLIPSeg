"""Tables, paired contrasts and the pre-specified gate. Reads saved cases only."""
import csv
from pathlib import Path

from .common import MODES, SEEDS, TARGETS, SMALL, OUTPUT, REPORT, RUNS, read, maybe, write_json, now, load_protocol, split_run
from .stats import load_cases, arm_summary, contrast, mean_sd

KEYS = [('DSC_percent', 'DSC'), ('small_DSC_percent', 'Small DSC'), ('NSD_percent', 'NSD'),
        ('micro_foreground_precision_percent', 'Precision'), ('micro_foreground_recall_percent', 'Recall')]


def analyse(cases, protocol, groups=None):
    """cases: {mode: {seed: rows}} for one cohort."""
    settings = protocol['statistics']
    arms = {}
    for mode in MODES:
        per_seed = {seed: arm_summary(cases[mode][seed], SMALL) for seed in SEEDS}
        summary = {}
        for key in list(per_seed[SEEDS[0]]):
            mean, sd = mean_sd([per_seed[s][key] for s in SEEDS])
            summary[key] = dict(mean=mean, sample_sd=sd, per_seed=[per_seed[s][key] for s in SEEDS])
        arms[mode] = summary
    contrasts = {}
    for lhs, rhs in protocol['contrasts']:
        contrasts[f'{lhs} - {rhs}'] = contrast(cases[lhs], cases[rhs], SMALL, settings['permutation_draws'],
                                               settings['bootstrap_draws'], settings['rng_seed'], groups)
    return dict(arms=arms, contrasts=contrasts)


def f(value, digits=2, sign=False):
    if value is None:
        return 'n/a'
    return f'{value:+.{digits}f}' if sign else f'{value:.{digits}f}'


def p(value):
    if value is None:
        return 'n/a'
    return '<0.001' if value < .001 else f'{value:.3f}'


def arm_table(result):
    lines = ['| Arm | DSC % | Small DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Zero overlap (per seed) |',
             '|---|---:|---:|---:|---:|---:|---|---|']
    for mode in MODES:
        a = result['arms'][mode]
        cells = [f"{f(a[k]['mean'])} ± {f(a[k]['sample_sd'])}" if a[k]['mean'] is not None else 'n/a' for k, _ in KEYS]
        lines.append(f"| {mode} | " + ' | '.join(cells) +
                     f" | {'/'.join(str(v) for v in a['positive_DSC_below_50_percent']['per_seed'])}"
                     f" | {'/'.join(str(v) for v in a['zero_overlap_positive']['per_seed'])} |")
    return lines


def contrast_table(result, subset, metric='DSC'):
    lines = ['| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |',
             '|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|']
    for name, c in result['contrasts'].items():
        d = c[metric][subset]
        if not d.get('n'):
            lines.append(f'| {name} | 0 |' + ' n/a |'*10)
            continue
        lines.append(f"| {name} | {d['n']} | {f(d['mean_pp'], 2, True)} | [{f(d['bootstrap_ci95_pp'][0], 2, True)}, {f(d['bootstrap_ci95_pp'][1], 2, True)}]"
                     f" | {' / '.join(f(v, 2, True) for v in d['per_seed_mean_pp'])} | {d['seeds_positive']}/{len(d['per_seed_mean_pp'])}"
                     f" | {d['improved']}/{d['worsened']}/{d['tied']} | {f(d['median_pp'], 2, True)} | {p(d['permutation_p'])} | {p(d['wilcoxon_p'])}"
                     f" | {f(d['top2_gain_contribution_to_mean_pp'], 2, True)} | {f(d['mean_of_others_without_top2_gains_pp'], 2, True)} |")
    return lines


def pixel_table(result):
    lines = ['| Contrast | Precision pp (per seed) | Recall pp (per seed) | FP pixels % (per seed) | FN pixels % (per seed) | DSC<50% count change |',
             '|---|---|---|---|---|---|']
    for name, c in result['contrasts'].items():
        x = c['pixels_and_failures_per_seed']
        row = lambda key, digits=2: ' / '.join(f(v, digits, True) for v in x[key])
        lines.append(f"| {name} | {row('micro_foreground_precision_percent_delta')} | {row('micro_foreground_recall_percent_delta')}"
                     f" | {row('fp_pixels_relative_change_percent', 1)} | {row('fn_pixels_relative_change_percent', 1)}"
                     f" | {' / '.join(f'{v:+d}' for v in x['positive_DSC_below_50_percent_delta'])} |")
    return lines


def section(title, result):
    lines = [f'## {title}', '', 'Three-seed mean ± sample SD.', '', *arm_table(result), '',
             '**Paired DSC differences, all images** (seed-averaged per image; tests are image-level and conditional on the three seeds).', '',
             *contrast_table(result, 'all'), '',
             '**Paired DSC differences, small targets only.**', '', *contrast_table(result, 'small'), '',
             '**Paired NSD differences, all images.**', '', *contrast_table(result, 'all', 'NSD'), '',
             '**Pixel-level and failure changes.**', '', *pixel_table(result), '']
    return lines


def evaluate_gate(results, protocol):
    gate = protocol['gate']
    key = ' - '.join(gate['primary_contrast'])
    dsc = {t: results[t]['contrasts'][key]['DSC']['all']['mean_pp'] for t in TARGETS}
    small = {t: results[t]['contrasts'][key]['DSC']['small']['mean_pp'] for t in TARGETS}
    ood_positive = [t for t in protocol['ood_targets'] if dsc[t] > 0]
    macro_small = sum(small.values())/len(small)
    checks = dict(
        g1_source_test_dsc_gain=dict(value_pp=dsc['Kvasir'], passed=dsc['Kvasir'] > gate['g1_source_test_dsc_gain_gt_pp']),
        g2_ood_sets_positive=dict(value=len(ood_positive), sets=ood_positive,
                                  passed=len(ood_positive) >= gate['g2_min_ood_sets_with_positive_dsc_gain']),
        g3_macro_small_dsc_gain=dict(value_pp=macro_small, per_set_pp=small,
                                     passed=macro_small > gate['g3_macro_small_dsc_gain_gt_pp']),
        guard_no_large_regression=dict(worst_pp=min(dsc.values()),
                                       passed=min(dsc.values()) >= -gate['guard_max_dsc_regression_any_set_pp']))
    return dict(primary_contrast=key, dsc_gain_pp=dsc, macro_dsc_gain_pp=sum(dsc.values())/len(dsc),
                macro_ood_dsc_gain_pp=sum(dsc[t] for t in protocol['ood_targets'])/len(protocol['ood_targets']),
                checks=checks, passed=all(c['passed'] for c in checks.values()))


def write_report():
    protocol = load_protocol()
    plan = read(OUTPUT/'evaluation_plan.json')
    results, rows = {}, []
    for target in TARGETS:
        cases = {mode: {} for mode in MODES}
        for run in RUNS:
            mode, seed = split_run(run)
            folder = OUTPUT/'evaluations'/run/target
            if not maybe(folder/'manifest.json').get('complete'):
                raise ValueError(f'Incomplete evaluation: {run}/{target}')
            cases[mode][seed] = load_cases(folder/'cases.csv')
            for r in cases[mode][seed]:
                rows.append(dict(dataset=target, arm=mode, seed=seed, **r))
        groups = list(plan['test_cohorts'][target]['image_hash_groups'].values())
        results[target] = analyse(cases, protocol, groups)
        results[target]['n_unique_images'] = len(groups)
    gate = evaluate_gate(results, protocol)
    checks = {run: read(OUTPUT/'evaluations'/run/'Val_check/comparison.json') for run in RUNS}
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', dict(protocol=protocol, protocol_sha256=plan['protocol_sha256'], gate=gate,
                                           results=results, val_reproduction=checks, generated_at=now(),
                                           independent_untouched_test=False))
    with (REPORT/'all_cases.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    c = gate['checks']
    mark = lambda ok: 'PASS' if ok else 'FAIL'
    lines = ['# Test/OOD evaluation of the detail and global_mix arms (generic prompt)', '',
             f"Suite `{protocol['name']}`; twelve accepted epoch-100 checkpoints; MC30, batch 32, threshold 0.5; fixed text `{protocol['prompt']}`.",
             'Evaluation only: no training, no checkpoint, threshold or prompt selection. The gate below was fixed before any of these results existed.', '',
             f"## Gate (detail - baseline): **{mark(gate['passed'])}**", '',
             '| Criterion | Value | Result |', '|---|---|---|',
             f"| G1 Kvasir Test DSC gain > 0 pp | {f(c['g1_source_test_dsc_gain']['value_pp'], 2, True)} pp | {mark(c['g1_source_test_dsc_gain']['passed'])} |",
             f"| G2 positive DSC gain on >= {protocol['gate']['g2_min_ood_sets_with_positive_dsc_gain']} of 4 OOD sets | {c['g2_ood_sets_positive']['value']} ({', '.join(c['g2_ood_sets_positive']['sets']) or 'none'}) | {mark(c['g2_ood_sets_positive']['passed'])} |",
             f"| G3 macro small-target DSC gain > 0 pp | {f(c['g3_macro_small_dsc_gain']['value_pp'], 2, True)} pp | {mark(c['g3_macro_small_dsc_gain']['passed'])} |",
             f"| Guard: no set worse than -{protocol['gate']['guard_max_dsc_regression_any_set_pp']} pp | worst {f(c['guard_no_large_regression']['worst_pp'], 2, True)} pp | {mark(c['guard_no_large_regression']['passed'])} |", '',
             '| Dataset | n | n small | detail - baseline DSC pp | Small DSC pp | Unique-image DSC pp |', '|---|---:|---:|---:|---:|---:|']
    for target in TARGETS:
        d = results[target]['contrasts'][gate['primary_contrast']]['DSC']
        lines.append(f"| {target} | {d['all']['n']} | {d['small'].get('n', 0)} | {f(d['all']['mean_pp'], 2, True)} | {f(d['small'].get('mean_pp'), 2, True)} | {f(d['unique_images']['mean_pp'], 2, True)} ({d['unique_images']['n']} images) |")
    lines += [f"| Macro mean, 5 sets | | | {f(gate['macro_dsc_gain_pp'], 2, True)} | {f(c['g3_macro_small_dsc_gain']['value_pp'], 2, True)} | |",
              f"| Macro mean, 4 OOD sets | | | {f(gate['macro_ood_dsc_gain_pp'], 2, True)} | | |", '']
    for target in TARGETS:
        lines += section(f"{target} Test" + (' (source domain)' if target == 'Kvasir' else ' (OOD)'), results[target])
    lines += ['## Val reproduction check on this machine', '',
              '| Run | Accepted Val DSC % | Local Val DSC % | Difference pp |', '|---|---:|---:|---:|']
    for run in RUNS:
        v = checks[run]
        lines.append(f"| {run} | {f(v['accepted_DSC_percent'], 4)} | {f(v['local_DSC_percent'], 4)} | {f(v['difference_pp'], 4, True)} |")
    lines += ['', '## Limitations', '', *[f'- {text}' for text in protocol['disclosures']],
              '- Three seeds; image-level tests are conditional on those seeds, not patient/video clustered, and not corrected for multiple comparisons.', '']
    (REPORT/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return REPORT/'report.md'
