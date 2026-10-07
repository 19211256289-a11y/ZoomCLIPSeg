"""Analysis of the box-crop control from saved per-image files. The t15 rows are the accepted two-pass results."""
import csv

import numpy as np

from .common import *


def rows_of(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def stats(diff, rng, draws, perms):
    diff = np.asarray(diff, float)
    boot = diff[rng.integers(0, len(diff), (draws, len(diff)))].mean(1)
    null = np.abs((rng.choice([-1., 1.], (perms, len(diff)))*diff).mean(1))
    return dict(mean_pp=float(diff.mean()), ci95_pp=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                p=float((1+np.sum(null >= abs(diff.mean())-1e-12))/(perms+1)), n=int(len(diff)))


def macro_stats(per_set, rng, draws, perms):
    values = [np.asarray(v, float) for v in per_set.values()]
    observed = float(np.mean([v.mean() for v in values]))
    boot = np.mean([v[rng.integers(0, len(v), (draws, len(v)))].mean(1) for v in values], 0)
    null = np.mean([(rng.choice([-1., 1.], (perms, len(v)))*v).mean(1) for v in values], 0)
    return dict(mean_pp=observed, ci95_pp=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                p=float((1+np.sum(np.abs(null) >= abs(observed)-1e-12))/(perms+1)))


def analyse(protocol):
    statistics = protocol['statistics']
    rng = np.random.default_rng(statistics['rng_seed'])
    draws, perms = statistics['bootstrap_draws'], statistics['permutation_draws']
    targets, ood, seeds = list(TARGETS), list(TARGETS)[1:], protocol['seeds']
    variants = [v['name'] for v in protocol['variants']]
    table, second, side = {}, {}, {}
    for run in protocol['runs']:
        seed = seed_of(run)
        for target in targets:
            mine = rows_of(OUTPUT/'evaluations'/run/target/'cases.csv')
            accepted = [r for r in rows_of(accepted_two_pass(protocol, run, target)) if r['config'] == 't15']
            for r in mine+accepted:
                if r['config'] != 'single' and r['config'] not in variants and r['config'] != 't15':
                    raise RuntimeError('Unexpected configuration: '+r['config'])
                table[(r['config'], target, seed, r['name'])] = 100*float(r['DSC'])
                second[(r['config'], target, seed, r['name'])] = r['kind'] == 'in'
                side[(r['config'], target, seed, r['name'])] = float(r['window'])
    names = {t: sorted({n for (c, tt, s, n) in table if c == 'single' and tt == t}) for t in targets}
    configs = ['single', 't15']+variants
    for config in configs:
        for t in targets:
            for s in seeds:
                if sorted(n for (c, tt, ss, n) in table if c == config and tt == t and ss == s) != names[t]:
                    raise RuntimeError(f'Cases do not match the cohort: {config}, {t}, seed {s}')
    result = {}
    for config in configs:
        mean = {t: float(np.mean([table[(config, t, s, n)] for s in seeds for n in names[t]])) for t in targets}
        zoomed = [side[(config, t, s, n)] for t in ood for s in seeds for n in names[t] if second[(config, t, s, n)]]
        block = dict(dsc=mean, ood_dsc=float(np.mean([mean[t] for t in ood])),
                     second_pass_share_ood=float(np.mean([np.mean([second[(config, t, s, n)] for s in seeds for n in names[t]]) for t in ood])),
                     median_window_side_ood=float(np.median(zoomed)) if zoomed else None)
        for label, other, sign in (('gain_over_single', 'single', 1), ('t15_minus_this', 't15', -1)):
            if config == other:
                continue
            per_set = {t: [sign*float(np.mean([table[(config, t, s, n)]-table[(other, t, s, n)] for s in seeds])) for n in names[t]]
                       for t in targets}
            per_seed = [sign*float(np.mean([np.mean([table[(config, t, s, n)]-table[(other, t, s, n)] for n in names[t]]) for t in ood]))
                        for s in seeds]
            block[label] = dict(sets={t: stats(per_set[t], rng, draws, perms) for t in targets},
                                ood_macro=macro_stats({t: per_set[t] for t in ood}, rng, draws, perms),
                                ood_macro_per_seed_pp=per_seed, seeds_positive=int(sum(v > 0 for v in per_seed)))
        result[config] = block
    verdicts = {}
    for config in variants:
        c = result[config]['t15_minus_this']['ood_macro']
        verdicts[config] = ('t15_better' if c['mean_pp'] > 0 and c['ci95_pp'][0] > 0 else
                            'variant_better' if c['mean_pp'] < 0 and c['ci95_pp'][1] < 0 else 'not_distinguishable')
    plain = [v for v in variants if v.startswith('plain')]
    overall = 'rule_adds_value' if all(verdicts[v] == 't15_better' for v in plain) else 'plain_crop_sufficient'
    return dict(methods=result, verdicts=verdicts, overall=overall)


def ci(s):
    return f"{s['mean_pp']:+.2f} [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}], p={s['p']:.4f}"


def write_report():
    protocol = load_protocol()
    analysis = analyse(protocol)
    plan = read(OUTPUT/'evaluation_plan.json')
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', dict(name=NAME, created_at=now(), plan_digest=plan['digest'], **analysis))
    variants = [v['name'] for v in protocol['variants']]
    lines = [f'# {NAME}: SCZ window rule against a plain crop of the predicted box (detail-branch model, five seeds)', '',
             'Nothing was trained. The locked configuration t15 remains the method whatever these numbers show.', '',
             'OOD = equal-weight mean of the four OOD polyp sets. Second pass = share of OOD images that receive a second pass; '
             'window = median window side of those images, as a fraction of the image side.', '',
             '| method | second pass | window | Kvasir DSC | OOD DSC | gain over single pass [95% CI], p | t15 minus method [95% CI], p | verdict |',
             '|---|---|---|---|---|---|---|---|']
    for config, block in analysis['methods'].items():
        gain = ci(block['gain_over_single']['ood_macro']) if 'gain_over_single' in block else ''
        versus = ci(block['t15_minus_this']['ood_macro']) if 't15_minus_this' in block else ''
        window = '' if block['median_window_side_ood'] is None else f"{block['median_window_side_ood']:.2f}"
        lines.append(f"| {config} | {100*block['second_pass_share_ood']:.0f}% | {window} | {block['dsc']['Kvasir']:.2f} | {block['ood_dsc']:.2f} | "
                     f"{gain} | {versus} | {analysis['verdicts'].get(config, '')} |")
    lines += ['', f"**Overall verdict by the rule fixed in advance: {analysis['overall']}.**", '',
              'Per test set, DSC (%):', '', '| method | '+' | '.join(TARGETS)+' |', '|---|'+'---|'*len(TARGETS)]
    for config, block in analysis['methods'].items():
        lines.append(f'| {config} | '+' | '.join(f"{block['dsc'][t]:.2f}" for t in TARGETS)+' |')
    lines += ['', 't15 minus method, per test set (pp):', '', '| method | '+' | '.join(TARGETS)+' |', '|---|'+'---|'*len(TARGETS)]
    for config in variants:
        sets = analysis['methods'][config]['t15_minus_this']['sets']
        lines.append(f'| {config} | '+' | '.join(ci(sets[t]) for t in TARGETS)+' |')
    lines += ['', 't15 minus method on the OOD mean, per seed (pp):', '']
    for config in variants:
        block = analysis['methods'][config]['t15_minus_this']
        lines.append(f"- {config}: {', '.join(f'{v:+.2f}' for v in block['ood_macro_per_seed_pp'])}")
    lines += ['', '## Definitions', '']+[f'- **{k}**: {v}' for k, v in protocol['window_rules'].items()]
    lines += [f"- **{v['name']}**: box rule with margin {v['margin']}, min_window {v['min_window']}" for v in protocol['variants']]
    lines += [f'- {k}: {v}' for k, v in protocol['variant_notes'].items()]
    lines += ['', '## Reading rules fixed in advance', '', f"- consequence: {protocol['reading_rules_fixed_in_advance']['consequence']}"]
    lines += ['', '## Disclosures', '']+[f'- {d}' for d in protocol['disclosures']]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return str(REPORT/'report.md')
