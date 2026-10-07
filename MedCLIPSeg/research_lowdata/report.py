"""Report of the reduced-training-set experiments from saved per-image files. The rows at 100% and all Polyp-PVT rows
are results of the same three seeds from other suites; they are read, not recomputed."""
import csv

import numpy as np

from research_polyp_pvt.report import load_cases, stats, macro_stats, mean_sd, ci, ms
from .common import *

ROWS = ('baseline', 'baseline_scz', 'detail', 'full', 'pvt')
LABELS = dict(baseline='Baseline (fixed sentence), single pass', baseline_scz='Baseline + SCZ',
              detail='Detail branch, single pass', full='ZoomCLIPSeg (detail branch + SCZ)', pvt='Polyp-PVT')
OURS = dict(baseline=('baseline', False), baseline_scz=('baseline', True), detail=('detail', False), full=('detail', True))
CONTRASTS = dict(full_minus_pvt=('full', 'pvt'), baseline_minus_pvt=('baseline', 'pvt'), full_minus_baseline=('full', 'baseline'),
                 scz_on_detail=('full', 'detail'), scz_on_baseline=('baseline_scz', 'baseline'), detail_branch=('detail', 'baseline'))
PVT_LOW = ROOT/'outputs_research/pvt_ext_v1'
FULL = 100


def complete(folder):
    return bool(maybe(folder/'manifest.json').get('complete'))


def source(suite, level, row, seed, target, two_pass):
    """(file, value of the configuration column or None) of one row, or None if that result does not exist."""
    if level == FULL:
        if row == 'pvt':
            path, config = ROOT/'outputs_research'/suite['reference']['full_data']['pvt']/'evaluations'/f'seed{seed}'/target/'cases.csv', None
        else:
            name, arm, config = suite['reference']['full_data'][row]
            path = ROOT/'outputs_research'/name/'evaluations'/f'{arm}_seed{seed}'/target/'cases.csv'
        return (path, config) if path.is_file() else None
    if row == 'pvt':
        folder = PVT_LOW/'evaluations'/f'kvasir_p{level}'/f'seed{seed}'/target
        return (folder/'cases.csv', None) if complete(folder) else None
    arm, second = OURS[row]
    folder = OUTPUT/'evaluations'/f'{arm}_p{level}_seed{seed}'/target
    return (folder/'cases.csv', two_pass if second else 'single') if complete(folder) else None


def second_pass_share(path, config):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        rows = [r for r in csv.DictReader(stream) if r['config'] == config]
    return float(np.mean([r['kind'] != 'none' for r in rows]))


def analyse_level(suite, level, two_pass):
    """Everything that exists for one training-set size. A row is used only if all its seeds and test sets exist."""
    seeds, targets, ood = suite['seeds'], suite['test_targets'], suite['ood_targets']
    statistics, threshold = suite['statistics'], suite['small_target_threshold']
    cases, shares = {}, {}
    for row in ROWS:
        found = {(s, t): source(suite, level, row, s, t, two_pass) for s in seeds for t in targets}
        if all(found.values()):
            cases[row] = {s: {t: load_cases(*found[s, t]) for t in targets} for s in seeds}
            if row in OURS and OURS[row][1]:
                shares[row] = float(np.mean([np.mean([second_pass_share(*found[s, t]) for s in seeds]) for t in ood]))
    if not cases:
        return None
    first = next(iter(cases.values()))
    names = {t: sorted(first[seeds[0]][t]) for t in targets}
    for row, block in cases.items():
        for s in seeds:
            for t in targets:
                if sorted(block[s][t]) != names[t]:
                    raise RuntimeError(f'Case list differs: {row}, {level}%, {t}, seed {s}')
    small = {t: [n for n in names[t] if 0 < first[seeds[0]][t][n]['gt_pixels'] and first[seeds[0]][t][n]['gt_fraction'] <= threshold]
             for t in targets}
    summary = {}
    for row, block in cases.items():
        entry = dict(sets={t: dict(n=len(names[t]),
                                   DSC=mean_sd([np.mean([block[s][t][n]['DSC'] for n in names[t]]) for s in seeds]),
                                   NSD=mean_sd([np.mean([block[s][t][n]['NSD'] for n in names[t]]) for s in seeds])) for t in targets})
        for key in ('DSC', 'NSD'):
            entry['OOD_'+key] = mean_sd([np.mean([np.mean([block[s][t][n][key] for n in names[t]]) for t in ood]) for s in seeds])
        entry['OOD_small_DSC'] = float(np.mean([block[s][t][n]['DSC'] for t in ood for s in seeds for n in small[t]]))
        entry['OOD_n_small'] = int(sum(len(small[t]) for t in ood))
        entry['OOD_missed_lesions_per_seed'] = [int(sum(block[s][t][n]['pred_pixels'] == 0 and block[s][t][n]['gt_pixels'] > 0
                                                        for t in ood for n in names[t])) for s in seeds]
        if row in shares:
            entry['OOD_second_pass_share'] = shares[row]
        summary[row] = entry
    contrasts = {}
    for index, (name, (a, b)) in enumerate(CONTRASTS.items()):
        if a not in cases or b not in cases:
            continue
        # One generator per contrast, so that an interval does not depend on which other rows happen to exist.
        rng = np.random.default_rng([statistics['rng_seed'], level, index])
        diff = {t: np.array([np.mean([cases[a][s][t][n]['DSC']-cases[b][s][t][n]['DSC'] for s in seeds]) for n in names[t]]) for t in targets}
        contrasts[name] = dict(
            minuend=a, subtrahend=b,
            sets={t: stats(diff[t], rng, statistics['bootstrap_draws'], statistics['permutation_draws']) for t in targets},
            ood_macro=macro_stats({t: diff[t] for t in ood}, rng, statistics['bootstrap_draws'], statistics['permutation_draws']),
            ood_macro_per_seed_pp=[float(np.mean([np.mean([cases[a][s][t][n]['DSC']-cases[b][s][t][n]['DSC'] for n in names[t]])
                                                  for t in ood])) for s in seeds])
    return dict(rows=list(cases), summary=summary, contrasts=contrasts)


def read_rules(levels, fractions):
    """The reading rules of the protocol, applied to what exists."""
    result = {}
    gap = {}
    for level in fractions+[FULL]:
        contrast = (levels.get(level) or {}).get('contrasts', {}).get('full_minus_pvt')
        if contrast is None:
            if level != FULL:
                result[f'p{level}'] = dict(verdict='not_available')
            continue
        gap[level] = -contrast['ood_macro']['mean_pp']       # advantage of Polyp-PVT over our full method
        if level != FULL:
            low, high = contrast['ood_macro']['ci95_pp']
            verdict = 'frozen_framework_ahead' if low > 0 else 'dedicated_network_ahead' if high < 0 else 'not_separated'
            result[f'p{level}'] = dict(verdict=verdict, full_minus_pvt=contrast['ood_macro'],
                                       follow_up_of_pvt_ext_triggered=bool(contrast['ood_macro']['mean_pp'] >= 0))
    first = fractions[0]
    if first in gap and FULL in gap:
        result['gap_narrows'] = dict(advantage_of_pvt_pp={f'p{level}': gap[level] for level in gap},
                                     reduction_pp=gap[FULL]-gap[first], holds=bool(gap[FULL]-gap[first] >= 1.))
    else:
        result['gap_narrows'] = dict(holds=None, note='not available')
    return result


def analyse(suite, partial):
    plan = read(OUTPUT/'experiment_plan.json')
    two_pass = plan['two_pass']['config']['name']
    fractions = list(suite['fractions'])
    levels = {level: analyse_level(suite, level, two_pass) for level in fractions+[FULL]}
    if not partial:
        for level in fractions:
            missing = [row for row in OURS if not levels[level] or row not in levels[level]['rows']]
            if missing:
                raise RuntimeError(f'Evaluations at {level}% are not complete: {missing}')
    queue = maybe(CONTROL/'queue.json').get('jobs', {})
    training = {}
    for fraction, arm, runs in groups(suite):
        records = [queue.get(f'train/{run}', {}) for run in runs]
        if all(r.get('status') == 'completed' for r in records):
            training[f'{arm}_p{fraction}'] = dict(minutes_per_seed=[r['active_seconds']/60 for r in records],
                                                  uninterrupted=[bool(r.get('uninterrupted')) for r in records])
    return dict(two_pass_config=two_pass, levels={str(k): v for k, v in levels.items() if v}, rules=read_rules(levels, fractions),
                training=training, lists=plan['lists'], plan_digest=plan['digest'])


def write_report(partial=False):
    suite = load_suite()
    analysis = analyse(suite, partial)
    targets, fractions = suite['test_targets'], list(suite['fractions'])
    levels, lists = analysis['levels'], analysis['lists']
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', dict(name=NAME, created_at=now(), partial=partial, **analysis))
    images = {str(f): lists[f'p{f}']['Train']['n'] for f in fractions}
    images[str(FULL)] = lists['full']['n']
    order = [str(f) for f in fractions]+[str(FULL)]
    lines = [f'# {NAME}: fixed-sentence models and Polyp-PVT trained on reduced Kvasir-SEG training sets', '',
             f"Three seeds ({', '.join(map(str, suite['seeds']))}), final-epoch weights, no selection on validation or test data. "
             'OOD = equal-weight mean of ColonDB, ClinicDB, CVC-300 and BKAI. Rows at 100% and all Polyp-PVT rows are read from '
             'the suites named in the protocol.', '']
    if partial:
        lines += ['**Partial report: only results that exist are shown; the reading rules are provisional.**', '']
    for key, title in (('OOD_DSC', 'OOD mean DSC (%)'), ('Kvasir', 'Kvasir test DSC (%, in distribution)')):
        lines += [f'## {title} by training-set size, mean +/- SD over seeds', '',
                  '| model | '+' | '.join(f'{level}% ({images[level]} images)' for level in order)+' |', '|---|'+'---|'*len(order)]
        for row in ROWS:
            cells = []
            for level in order:
                block = levels.get(level, {}).get('summary', {}).get(row)
                cells.append('' if block is None else ms(block[key] if key == 'OOD_DSC' else block['sets'][targets[0]]['DSC']))
            if any(cells):
                lines.append(f'| {LABELS[row]} | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['## Reading rules fixed in advance', '']
    for fraction in fractions:
        rule = analysis['rules'][f'p{fraction}']
        if rule['verdict'] == 'not_available':
            lines.append(f'- {fraction}%: not available (our full method or Polyp-PVT on this subset is not complete).')
        else:
            lines.append(f"- {fraction}%: **{rule['verdict']}** (ZoomCLIPSeg minus Polyp-PVT, OOD mean: {ci(rule['full_minus_pvt'])}). "
                         f"Conditional follow-up of pvt_ext_v1 triggered: {rule['follow_up_of_pvt_ext_triggered']}.")
    narrows = analysis['rules']['gap_narrows']
    if narrows['holds'] is None:
        lines.append('- gap_narrows: not available.')
    else:
        lines.append('- gap_narrows: **'+str(narrows['holds'])+'** (advantage of Polyp-PVT over ZoomCLIPSeg on the OOD mean: '
                     + ', '.join(f"{value:+.2f} pp at {level[1:]}%" for level, value in narrows['advantage_of_pvt_pp'].items())
                     + f"; reduction from 100% to {fractions[0]}%: {narrows['reduction_pp']:+.2f} pp, rule: at least 1 pp).")
    lines.append('')
    for level in order:
        block = levels.get(level)
        if not block:
            continue
        lines += [f'## Training on {level}% ({images[level]} images): DSC (%) per test set, mean +/- SD over seeds', '',
                  '| model | '+' | '.join(targets)+' | OOD mean | OOD NSD | small lesions, OOD pooled | lesion images without prediction, OOD (per seed) | second pass (share of OOD images) |',
                  '|---|'+'---|'*(len(targets)+5)]
        for row in block['rows']:
            s = block['summary'][row]
            share = f"{100*s['OOD_second_pass_share']:.0f}%" if 'OOD_second_pass_share' in s else ''
            lines.append(f'| {LABELS[row]} | '+' | '.join(ms(s['sets'][t]['DSC']) for t in targets)
                         + f" | {ms(s['OOD_DSC'])} | {ms(s['OOD_NSD'])} | {s['OOD_small_DSC']:.2f} (n={s['OOD_n_small']}) | "
                         + '/'.join(map(str, s['OOD_missed_lesions_per_seed']))+f' | {share} |')
        lines.append('')
        if block['contrasts']:
            lines += ['Differences in DSC, percentage points [95% CI], p:', '',
                      '| contrast | '+' | '.join(targets)+' | OOD mean | OOD mean per seed |', '|---|'+'---|'*(len(targets)+2)]
            for name, c in block['contrasts'].items():
                lines.append(f"| {name} ({LABELS[c['minuend']]} minus {LABELS[c['subtrahend']]}) | "
                             + ' | '.join(ci(c['sets'][t]) for t in targets)+f" | {ci(c['ood_macro'])} | "
                             + ', '.join(f'{v:+.2f}' for v in c['ood_macro_per_seed_pp'])+' |')
            lines.append('')
    lines += ['## Training', '']
    for fraction in fractions:
        steps = lists[f'p{fraction}']['optimisation_steps']
        lines.append(f"- {fraction}%: {lists[f'p{fraction}']['Train']['n']} training images, {steps['per_epoch']} mini-batches of "
                     f"{steps['batch_size']} per epoch, {steps['total']} optimisation steps in {suite['epochs']} epochs "
                     f"(full training set: {lists['full']['optimisation_steps']['total']}).")
    for name, record in analysis['training'].items():
        lines.append(f"- {name}: training time per seed {', '.join(f'{m:.0f} min' for m in record['minutes_per_seed'])}"
                     + ('' if all(record['uninterrupted']) else ' (at least one run was paused and continued)')+'.')
    lines += ['', '## Consequence fixed in advance', '', suite['reading_rules_fixed_in_advance']['consequence'].strip(),
              '', '## Disclosures', '']+[f'- {d}' for d in suite['disclosures']]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return str(REPORT/'report.md')
