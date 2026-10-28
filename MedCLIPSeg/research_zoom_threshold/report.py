"""Threshold sensitivity of SCZ from saved per-image files. The t15 rows (accepted SCZ at threshold 0.5) are read from
the accepted results and serve as a reproduction check of scz_t50."""
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
    names = {t: sorted({n for (c, tt, s, n) in table if c == REPRODUCTION_CONFIG and tt == t}) for t in targets}
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
        share = float(np.mean([np.mean([second[(config, t, s, n)] for s in seeds for n in names[t]]) for t in ood]))
        mean = {t: float(np.mean([table[(config, t, s, n)] for s in seeds for n in names[t]])) for t in targets}
        methods[config] = dict(threshold=threshold_of(config), second_pass_share_ood=share, dsc=mean,
                               ood_dsc=float(np.mean([mean[t] for t in ood])))
    gains = {TAG[t]: difference(f'scz_t{TAG[t]}', f'single_t{TAG[t]}') for t in THRESHOLDS}
    versus_half = {TAG[t]: dict(single=difference(f'single_t{TAG[t]}', 'single_t50'), scz=difference(f'scz_t{TAG[t]}', 'scz_t50'))
                   for t in THRESHOLDS if t != .5}
    reproduction = difference('scz_t50', 't15')
    robust = all(g['ood_macro']['mean_pp'] > 0 and g['ood_macro']['ci95_pp'][0] > 0 for g in gains.values())
    verdicts = dict(scz_gain_robust_to_threshold=bool(robust),
                    thresholds_with_positive_gain=[k for k, g in gains.items() if g['ood_macro']['mean_pp'] > 0],
                    thresholds_with_interval_above_zero=[k for k, g in gains.items() if g['ood_macro']['ci95_pp'][0] > 0],
                    scz_t50_reproduces_t15=bool(abs(reproduction['ood_macro']['mean_pp']) <= protocol['reproduction_tolerance_pp']))
    return dict(methods=methods, scz_gain_by_threshold=gains, versus_threshold_half=versus_half,
                reproduction_scz_t50_minus_t15=reproduction, verdicts=verdicts)


def ci(s):
    return f"{s['mean_pp']:+.2f} [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}], p={s['p']:.4f}"


def write_report():
    protocol = load_protocol()
    analysis = analyse(protocol)
    plan = read(OUTPUT/'evaluation_plan.json')
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', dict(name=NAME, created_at=now(), plan_digest=plan['digest'], **analysis))
    methods, gains, verdicts = analysis['methods'], analysis['scz_gain_by_threshold'], analysis['verdicts']
    rules = protocol['reading_rules_fixed_in_advance']
    lines = [f'# {NAME}: sensitivity of SCZ to the binarisation threshold (detail-branch model, five seeds)', '',
             'Nothing was trained. The locked method (threshold 0.5) is unchanged whatever these numbers show. ETIS-Larib is not touched.', '',
             '| threshold | single pass: Kvasir / OOD | SCZ: Kvasir / OOD | second pass (OOD) | SCZ gain, OOD mean [95% CI], p | per seed |',
             '|---|---|---|---|---|---|']
    for t in THRESHOLDS:
        a, b, g = methods[f'single_t{TAG[t]}'], methods[f'scz_t{TAG[t]}'], gains[TAG[t]]
        lines.append(f"| {t:.1f} | {a['dsc']['Kvasir']:.2f} / {a['ood_dsc']:.2f} | {b['dsc']['Kvasir']:.2f} / {b['ood_dsc']:.2f} | "
                     f"{100*b['second_pass_share_ood']:.0f}% | {ci(g['ood_macro'])} | "
                     f"{', '.join(f'{v:+.2f}' for v in g['ood_macro_per_seed_pp'])} |")
    rep = analysis['reproduction_scz_t50_minus_t15']
    lines += ['', f"Reproduction: scz_t50 minus the accepted t15, OOD mean {ci(rep['ood_macro'])} "
                  f"(tolerance {protocol['reproduction_tolerance_pp']} pp): **{verdicts['scz_t50_reproduces_t15']}**", '',
              '## Verdict by the rule fixed in advance', '',
              f"- {rules['robust']}: **{verdicts['scz_gain_robust_to_threshold']}**",
              f"- thresholds with a positive mean gain: {', '.join(verdicts['thresholds_with_positive_gain']) or 'none'}; "
              f"with the interval above zero: {', '.join(verdicts['thresholds_with_interval_above_zero']) or 'none'}", '',
              '## Change against threshold 0.5 (OOD mean, pp)', '', '| threshold | single pass | SCZ |', '|---|---|---|']
    for key, block in analysis['versus_threshold_half'].items():
        lines.append(f"| 0.{key[0]} | {ci(block['single']['ood_macro'])} | {ci(block['scz']['ood_macro'])} |")
    lines += ['', 'Per test set, DSC (%):', '', '| configuration | '+' | '.join(TARGETS)+' |', '|---|'+'---|'*len(TARGETS)]
    for config in list(CONFIGS)+['t15']:
        lines.append(f'| {config} | '+' | '.join(f"{methods[config]['dsc'][t]:.2f}" for t in TARGETS)+' |')
    lines += ['', '## Definitions', '']+[f'- **{k}**: {v}' for k, v in protocol['definitions'].items()]
    lines += ['', f"Consequence fixed in advance: {rules['consequence']}"]
    lines += ['', '## Disclosures', '']+[f'- {d}' for d in protocol['disclosures']]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return str(REPORT/'report.md')
