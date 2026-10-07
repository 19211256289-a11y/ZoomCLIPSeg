"""Analysis of the comparison from saved per-image files. The t15 rows are the accepted two-pass results."""
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
    table, second = {}, {}
    for run in protocol['runs']:
        seed = seed_of(run)
        for target in targets:
            for r in rows_of(OUTPUT/'evaluations'/run/target/'cases.csv'):
                table[(r['config'], target, seed, r['name'])] = 100*float(r['DSC'])
            accepted = [r for r in rows_of(accepted_two_pass(protocol, run, target)) if r['config'] == 't15']
            for r in accepted:
                table[('t15', target, seed, r['name'])] = 100*float(r['DSC'])
                second[(target, seed, r['name'])] = r['kind'] == 'in'
    names = {t: sorted({n for (c, tt, s, n) in table if c == 'single' and tt == t}) for t in targets}
    for t in targets:
        for s in seeds:
            if sorted(n for (c, tt, ss, n) in table if c == 't15' and tt == t and ss == s) != names[t]:
                raise RuntimeError(f'Accepted two-pass cases do not match the cohort: {t}, seed {s}')
    cost = {name: float(spec['runs_per_image']) for name, spec in protocol['alternatives'].items()}
    cost['single'] = 1.
    cost['t15'] = 1+float(np.mean([np.mean([second[(t, s, n)] for s in seeds for n in names[t]]) for t in ood]))
    result = {}
    for config in ('single', 't15')+ALTERNATIVES:
        mean = {t: float(np.mean([table[(config, t, s, n)] for s in seeds for n in names[t]])) for t in targets}
        block = dict(runs_per_image_ood=cost[config], dsc=mean, ood_dsc=float(np.mean([mean[t] for t in ood])))
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
    for config in ALTERNATIVES:
        c = result[config]['t15_minus_this']['ood_macro']
        dominates = c['mean_pp'] < 0 and c['ci95_pp'][1] < 0 and cost[config] <= cost['t15']
        preferable = c['mean_pp'] >= -.5 and cost['t15'] < cost[config]
        verdicts[config] = ('alternative_dominates_t15' if dominates else 't15_preferable' if preferable else
                            'alternative_more_accurate_at_higher_cost' if c['mean_pp'] < -.5 else 'undecided')
    return dict(methods=result, verdicts=verdicts)


def ci(s):
    return f"{s['mean_pp']:+.2f} [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}], p={s['p']:.4f}"


def write_report():
    protocol = load_protocol()
    analysis = analyse(protocol)
    plan = read(OUTPUT/'evaluation_plan.json')
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', dict(name=NAME, created_at=now(), plan_digest=plan['digest'], **analysis))
    lines = [f'# {NAME}: SCZ against generic inference alternatives (detail-branch model, five seeds)', '',
             'Nothing was trained. The locked configuration t15 remains the method whatever these numbers show.', '',
             'OOD = equal-weight mean of the four OOD polyp sets; runs = full inference runs per image (30 stochastic passes each).', '',
             '| method | runs per image | Kvasir DSC | OOD DSC | gain over single pass [95% CI], p | t15 minus method [95% CI], p | verdict |',
             '|---|---|---|---|---|---|---|']
    for config, block in analysis['methods'].items():
        gain = ci(block['gain_over_single']['ood_macro']) if 'gain_over_single' in block else ''
        versus = ci(block['t15_minus_this']['ood_macro']) if 't15_minus_this' in block else ''
        lines.append(f"| {config} | {block['runs_per_image_ood']:.2f} | {block['dsc']['Kvasir']:.2f} | {block['ood_dsc']:.2f} | {gain} | {versus} | "
                     f"{analysis['verdicts'].get(config, '')} |")
    lines += ['', 'Per test set, DSC (%):', '', '| method | '+' | '.join(TARGETS)+' |', '|---|'+'---|'*len(TARGETS)]
    for config, block in analysis['methods'].items():
        lines.append(f'| {config} | '+' | '.join(f"{block['dsc'][t]:.2f}" for t in TARGETS)+' |')
    lines += ['', 't15 minus method on the OOD mean, per seed (pp):', '']
    for config in ALTERNATIVES:
        block = analysis['methods'][config]['t15_minus_this']
        lines.append(f"- {config}: {', '.join(f'{v:+.2f}' for v in block['ood_macro_per_seed_pp'])}")
    lines += ['', '## Definitions', '']+[f"- **{k}** ({v['runs_per_image']} runs): {v['definition']}" for k, v in protocol['alternatives'].items()]
    lines += ['', '## Disclosures', '']+[f'- {d}' for d in protocol['disclosures']]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return str(REPORT/'report.md')
