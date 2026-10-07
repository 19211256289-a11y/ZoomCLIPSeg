"""Analysis of accuracy against inference budget from saved per-image files. The t15 rows (SCZ, 30 passes) are the
accepted two-pass results."""
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
    configs = list(CONFIGS)+['t15']
    table, second = {}, {}
    for run in protocol['runs']:
        seed = seed_of(run)
        for target in targets:
            mine = rows_of(OUTPUT/'evaluations'/run/target/'cases.csv')
            accepted = [r for r in rows_of(accepted_two_pass(protocol, run, target)) if r['config'] == 't15']
            for r in mine+accepted:
                if r['config'] not in configs:
                    raise RuntimeError('Unexpected configuration: '+r['config'])
                table[(r['config'], target, seed, r['name'])] = 100*float(r['DSC'])
                second[(r['config'], target, seed, r['name'])] = r['kind'] == 'in'
    names = {t: sorted({n for (c, tt, s, n) in table if c == 'single30' and tt == t}) for t in targets}
    for config in configs:
        for t in targets:
            for s in seeds:
                if sorted(n for (c, tt, ss, n) in table if c == config and tt == t and ss == s) != names[t]:
                    raise RuntimeError(f'Cases do not match the cohort: {config}, {t}, seed {s}')

    def difference(a, b):
        per_set = {t: [float(np.mean([table[(a, t, s, n)]-table[(b, t, s, n)] for s in seeds])) for n in names[t]] for t in targets}
        per_seed = [float(np.mean([np.mean([table[(a, t, s, n)]-table[(b, t, s, n)] for n in names[t]]) for t in ood])) for s in seeds]
        return dict(sets={t: stats(per_set[t], rng, draws, perms) for t in targets},
                    ood_macro=macro_stats({t: per_set[t] for t in ood}, rng, draws, perms),
                    ood_macro_per_seed_pp=per_seed, seeds_positive=int(sum(v > 0 for v in per_seed)))

    methods = {}
    for config in configs:
        k = passes_of(config)
        share = float(np.mean([np.mean([second[(config, t, s, n)] for s in seeds for n in names[t]]) for t in ood]))
        factor = 2 if config in HFLIP+SCZFLIP else 1
        mean = {t: float(np.mean([table[(config, t, s, n)] for s in seeds for n in names[t]])) for t in targets}
        methods[config] = dict(passes_per_run=k, second_pass_share_ood=share, forward_passes_per_image_ood=factor*k*(1+share),
                               dsc=mean, ood_dsc=float(np.mean([mean[t] for t in ood])))
        if config != 'single30':
            methods[config]['minus_single30'] = difference(config, 'single30')
    contrasts = {name: difference(a, b) for name, (a, b) in protocol['contrasts'].items()}
    positive = lambda name: contrasts[name]['ood_macro']['mean_pp'] > 0 and contrasts[name]['ood_macro']['ci95_pp'][0] > 0
    enough = [k for k in PASSES if k == 30 or methods[f'single{k:02d}']['minus_single30']['ood_macro']['mean_pp'] >= -.2]
    verdicts = dict(complementary=bool(positive('scz_adds_to_reflection') and positive('reflection_adds_to_scz')),
                    scz_adds_to_reflection=bool(positive('scz_adds_to_reflection')),
                    reflection_adds_to_scz=bool(positive('reflection_adds_to_scz')),
                    scz_at_no_extra_cost=bool(positive('scz_at_standard_budget')), passes_sufficient=int(min(enough)))
    return dict(methods=methods, contrasts=contrasts, verdicts=verdicts)


def ci(s):
    return f"{s['mean_pp']:+.2f} [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}], p={s['p']:.4f}"


LABELS = dict(single='single pass', hflip='reflection averaging', scz='SCZ', sczflip='SCZ + reflection averaging', t15='SCZ (locked method)')


def label_of(config):
    return LABELS['t15'] if config == 't15' else f"{LABELS[config[:-2]]}, {passes_of(config)} passes"


def write_report():
    protocol = load_protocol()
    analysis = analyse(protocol)
    plan = read(OUTPUT/'evaluation_plan.json')
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', dict(name=NAME, created_at=now(), plan_digest=plan['digest'], **analysis))
    methods, contrasts, verdicts = analysis['methods'], analysis['contrasts'], analysis['verdicts']
    lines = [f'# {NAME}: accuracy against inference budget (detail-branch model, five seeds)', '',
             'Nothing was trained. The locked method (t15 with 30 passes) is unchanged whatever these numbers show.', '',
             'OOD = equal-weight mean of the four OOD polyp sets. Forward passes = mean number of stochastic forward passes per OOD image; '
             'the standard single pass uses 30.', '',
             '| configuration | forward passes | second pass | Kvasir DSC | OOD DSC | minus the standard single pass [95% CI], p |',
             '|---|---|---|---|---|---|']
    for config in sorted(methods, key=lambda c: methods[c]['forward_passes_per_image_ood']):
        block = methods[config]
        delta = ci(block['minus_single30']['ood_macro']) if 'minus_single30' in block else ''
        lines.append(f"| {label_of(config)} | {block['forward_passes_per_image_ood']:.1f} | {100*block['second_pass_share_ood']:.0f}% | "
                     f"{block['dsc']['Kvasir']:.2f} | {block['ood_dsc']:.2f} | {delta} |")
    lines += ['', '## Contrasts fixed in advance (OOD mean, pp)', '', '| contrast | difference [95% CI], p | per seed |', '|---|---|---|']
    for name, (a, b) in protocol['contrasts'].items():
        block = contrasts[name]
        lines.append(f"| {name}: {a} minus {b} | {ci(block['ood_macro'])} | {', '.join(f'{v:+.2f}' for v in block['ood_macro_per_seed_pp'])} |")
    lines += ['', '## Verdicts by the rules fixed in advance', '',
              f"- SCZ adds to reflection averaging: **{verdicts['scz_adds_to_reflection']}**; reflection averaging adds to SCZ: "
              f"**{verdicts['reflection_adds_to_scz']}**; complementary (both): **{verdicts['complementary']}**",
              f"- SCZ with 15 passes per run is better than the standard single pass with 30 (at a lower number of forward passes): **{verdicts['scz_at_no_extra_cost']}**",
              f"- Smallest number of passes within 0.2 pp of 30 passes (single pass, OOD mean): **{verdicts['passes_sufficient']}**", '',
              'Per test set, DSC (%):', '', '| configuration | '+' | '.join(TARGETS)+' |', '|---|'+'---|'*len(TARGETS)]
    for config in sorted(methods, key=lambda c: methods[c]['forward_passes_per_image_ood']):
        lines.append(f'| {label_of(config)} | '+' | '.join(f"{methods[config]['dsc'][t]:.2f}" for t in TARGETS)+' |')
    lines += ['', '## Definitions', '']+[f'- **{k}**: {v}' for k, v in protocol['definitions'].items()]
    lines += ['', f"Consequence fixed in advance: {protocol['reading_rules_fixed_in_advance']['consequence']}"]
    lines += ['', '## Disclosures', '']+[f'- {d}' for d in protocol['disclosures']]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return str(REPORT/'report.md')
