"""Tables and the pre-specified contrasts of the control experiments. Works on whatever arms are complete. numpy only."""
import csv

import numpy as np

from research_detail_test.stats import FLOATS, INTS, contrast
from .common import OUTPUT, REPORT, load_suite, maybe, write_json, now


def load(path):
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as stream:
        for r in csv.DictReader(stream):
            row = dict(config=r['config'], name=r['name'], image_name=r['image_name'], kind=r['kind'])
            for k in INTS:
                row[k] = int(float(r[k]))
            for k in FLOATS:
                row[k] = float(r[k]) if r[k] not in ('', 'None') else None
            rows.append(row)
    return rows


def collect(suite):
    """{arm: {cohort: {seed: rows}}} for arms whose three seeds are completely evaluated."""
    cohorts, result = ['Val']+suite['test_targets'], {}
    for arm in suite['arms']:
        data = {c: {} for c in cohorts}
        for seed in suite['seeds']:
            for cohort in cohorts:
                folder = OUTPUT/'evaluations'/f'{arm}_seed{seed}'/cohort
                if maybe(folder/'manifest.json').get('complete'):
                    data[cohort][seed] = load(folder/'cases.csv')
        if all(len(data[c]) == len(suite['seeds']) for c in cohorts):
            result[arm] = data
    return result


def analyse(data, suite):
    targets, ood, seeds, small, stat = suite['test_targets'], suite['ood_targets'], suite['seeds'], suite['small_target_threshold'], suite['statistics']
    pick = lambda arm, cohort, config: {s: [r for r in data[arm][cohort][s] if r['config'] == config] for s in seeds}
    per_seed = lambda rows, metric='DSC': [100*float(np.mean([r[metric] for r in rows[s]])) for s in seeds]
    arms = {}
    for arm in data:
        configs = ['single']+(['t15'] if suite['arms'][arm]['two_pass'] else [])
        arms[arm] = {}
        for config in configs:
            entry = {}
            for cohort in ['Val']+targets:
                if cohort == 'Val' and config != 'single':
                    continue
                rows = pick(arm, cohort, config)
                entry[cohort] = dict(DSC=per_seed(rows), NSD=per_seed(rows, 'NSD'))
            entry['OOD mean'] = dict(DSC=[float(np.mean([entry[t]['DSC'][i] for t in ood])) for i in range(len(seeds))])
            arms[arm][config] = entry
    contrasts = {}
    for name, (a, ca, b, cb) in suite['contrasts'].items():
        if a not in data or b not in data:
            continue
        per_target = {}
        for target in targets:
            c = contrast(pick(a, target, ca), pick(b, target, cb), small, stat['permutation_draws'], stat['bootstrap_draws'], stat['rng_seed'])
            per_target[target] = dict(all=c['DSC']['all'], small=c['DSC']['small'], NSD=c['NSD']['all'])
        macro_seed = np.mean([per_target[t]['all']['per_seed_mean_pp'] for t in ood], axis=0)
        all_seed = np.mean([per_target[t]['all']['per_seed_mean_pp'] for t in targets], axis=0)
        contrasts[name] = dict(lhs=[a, ca], rhs=[b, cb], targets=per_target, macro_ood_pp=float(macro_seed.mean()),
                               macro_ood_per_seed_pp=[float(v) for v in macro_seed], macro_all_pp=float(all_seed.mean()),
                               macro_all_per_seed_pp=[float(v) for v in all_seed])
    readings = {}
    c = contrasts.get('two_pass_on_detail_scale')
    if c:
        gain, positive = c['macro_ood_pp'], all(v > 0 for v in c['macro_ood_per_seed_pp'])
        readings['two_pass_after_scale_augmentation'] = 'retained' if gain >= 1.0 and positive else 'absorbed' if gain < 0.5 else 'reduced'
    c = contrasts.get('true_448_vs_upsampled')
    if c:
        readings['detail_information'] = 'supported' if c['macro_all_pp'] > 0 and sum(v > 0 for v in c['macro_all_per_seed_pp']) >= 2 else 'not supported'
    return dict(arms=arms, contrasts=contrasts, readings=readings, seeds=seeds)


def write_report():
    suite = load_suite()
    data = collect(suite)
    if not data:
        return None
    result = analyse(data, suite)
    result.update(name=suite['name'], created_at=now(), arms_complete=list(data), arms_pending=[a for a in suite['arms'] if a not in data])
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', result)
    targets = suite['test_targets']
    ms = lambda v: f'{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}'
    lines = [f"# {suite['name']}: control experiments (Kvasir source, trained on this PC)", '',
             f"Seeds {', '.join(map(str, suite['seeds']))}; 100 epochs; fixed sentence; MC30; final-epoch weights.",
             f"Arms complete: {', '.join(data)}." + (f" Pending: {', '.join(result['arms_pending'])}." if result['arms_pending'] else ''), '',
             '## DSC (%), mean ± SD over seeds', '', '| arm | inference | Val | ' + ' | '.join(targets) + ' | OOD mean |', '|---|---|---|' + '---|'*(len(targets)+1)]
    for arm in data:
        for config, entry in result['arms'][arm].items():
            val = ms(entry['Val']['DSC']) if 'Val' in entry else ''
            lines.append(f"| {arm} | {'two-pass' if config == 't15' else 'single'} | {val} | " + ' | '.join(ms(entry[t]['DSC']) for t in targets)
                         + f" | {ms(entry['OOD mean']['DSC'])} |")
    lines += ['', '## Pre-specified contrasts (paired DSC difference in pp; 95% bootstrap interval over images)', '',
              '| contrast | definition | ' + ' | '.join(targets) + ' | OOD mean | per seed |', '|---|---|' + '---|'*(len(targets)+2)]
    label = lambda pair: pair[0] + (' two-pass' if pair[1] == 't15' else '')
    for name, c in result['contrasts'].items():
        cells = [f"{c['targets'][t]['all']['mean_pp']:+.2f} [{c['targets'][t]['all']['bootstrap_ci95_pp'][0]:+.2f}, {c['targets'][t]['all']['bootstrap_ci95_pp'][1]:+.2f}]" for t in targets]
        lines.append(f"| {name} | {label(c['lhs'])} − {label(c['rhs'])} | " + ' | '.join(cells) + f" | {c['macro_ood_pp']:+.2f} | "
                     + ', '.join(f'{v:+.2f}' for v in c['macro_ood_per_seed_pp']) + ' |')
    lines += ['', '## Readings fixed in advance', '']
    if 'two_pass_after_scale_augmentation' in result['readings']:
        c = result['contrasts']['two_pass_on_detail_scale']
        lines.append(f"- Two-pass inference after scale augmentation: **{result['readings']['two_pass_after_scale_augmentation']}** "
                     f"(macro OOD gain {c['macro_ood_pp']:+.2f} pp; retained >= 1.0 and every seed positive, absorbed < 0.5).")
    if 'detail_information' in result['readings']:
        c = result['contrasts']['true_448_vs_upsampled']
        lines.append(f"- Detail branch needs the real 448 image: **{result['readings']['detail_information']}** "
                     f"(mean over five sets {c['macro_all_pp']:+.2f} pp; per seed {', '.join(f'{v:+.2f}' for v in c['macro_all_per_seed_pp'])}).")
    if not result['readings']:
        lines.append('- Not available yet.')
    lines += ['', 'Small-lesion and NSD differences, permutation p-values and per-seed counts are in results.json.',
              'Disclosures: ' + ' '.join(suite['disclosures']), '']
    (REPORT/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return REPORT/'report.md'
