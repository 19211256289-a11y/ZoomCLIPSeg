"""Analysis of SCZ with one stochastic pass from saved per-image files. The t15 rows (SCZ, 30 passes) are the accepted
two-pass results."""
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
        share = float(np.mean([np.mean([second[(config, t, s, n)] for s in seeds for n in names[t]]) for t in ood]))
        mean = {t: float(np.mean([table[(config, t, s, n)] for s in seeds for n in names[t]])) for t in targets}
        per_seed = [float(np.mean([np.mean([table[(config, t, s, n)] for n in names[t]]) for t in ood])) for s in seeds]
        methods[config] = dict(passes_per_run=passes_of(config), second_pass_share_ood=share, dsc=mean,
                               ood_dsc=float(np.mean([mean[t] for t in ood])), ood_dsc_per_seed=per_seed)
    contrasts = {name: difference(a, b) for name, (a, b) in protocol['contrasts'].items()}
    rules = protocol['reading_rules_fixed_in_advance']
    margin = rules['non_inferiority_margin_pp']
    ni = contrasts['scz01_vs_t15']['ood_macro']
    gain = contrasts['scz01_vs_single01']
    verdicts = dict(one_pass_non_inferior=bool(ni['ci95_pp'][0] > -margin),
                    scz_gain_with_one_pass=bool(gain['seeds_positive'] == len(seeds) and gain['ood_macro']['ci95_pp'][0] > 0))
    return dict(methods=methods, contrasts=contrasts, verdicts=verdicts)


def ci(s):
    return f"{s['mean_pp']:+.2f} [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}], p={s['p']:.4f}"


LABELS = dict(single30='single pass, 30 stochastic passes (accepted)', single01='single pass, 1 stochastic pass',
              scz01='SCZ, 1 stochastic pass per run', t15='SCZ, 30 stochastic passes per run (locked method)')


def write_report():
    protocol = load_protocol()
    analysis = analyse(protocol)
    plan = read(OUTPUT/'evaluation_plan.json')
    timing = maybe(OUTPUT/'timing.json')
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', dict(name=NAME, created_at=now(), plan_digest=plan['digest'], timing=timing, **analysis))
    methods, contrasts, verdicts = analysis['methods'], analysis['contrasts'], analysis['verdicts']
    rules = protocol['reading_rules_fixed_in_advance']
    lines = [f'# {NAME}: SCZ with one stochastic pass per run (detail-branch model, five seeds)', '',
             'Nothing was trained. The locked method (t15) is unchanged whatever these numbers show. ETIS-Larib is not touched.', '',
             '| configuration | second pass (OOD) | Kvasir DSC | OOD DSC | OOD per seed |', '|---|---|---|---|---|']
    for config in ('single30', 'single01', 'scz01', 't15'):
        block = methods[config]
        lines.append(f"| {LABELS[config]} | {100*block['second_pass_share_ood']:.0f}% | {block['dsc']['Kvasir']:.2f} | "
                     f"{block['ood_dsc']:.2f} | {', '.join(f'{v:.2f}' for v in block['ood_dsc_per_seed'])} |")
    lines += ['', '## Contrasts fixed in advance (pp)', '',
              '| contrast | Kvasir | OOD mean [95% CI], p | per seed |', '|---|---|---|---|']
    for name, (a, b) in protocol['contrasts'].items():
        block = contrasts[name]
        lines.append(f"| {name}: {a} minus {b} | {ci(block['sets']['Kvasir'])} | {ci(block['ood_macro'])} | "
                     f"{', '.join(f'{v:+.2f}' for v in block['ood_macro_per_seed_pp'])} |")
    lines += ['', '## Verdicts by the rules fixed in advance', '',
              f"- {rules['one_pass_non_inferior']}: **{verdicts['one_pass_non_inferior']}**",
              f"- {rules['scz_gain_with_one_pass']}: **{verdicts['scz_gain_with_one_pass']}**", '']
    if timing:
        lines += [f"## Time per image at batch size 1 ({timing['device']}, {timing['dataset']}, {timing['run']}, "
                  f"{timing['measured_images']} images after {timing['warmup_images']} warm-up images)", '',
                  '| setting | mean ms (over repetitions) | range of repetition means | second pass share |', '|---|---|---|---|']
        for label, block in timing['settings'].items():
            share = np.mean([r['second_pass_share'] for r in block['repetitions']])
            lines.append(f"| {LABELS[label]} | {block['mean_ms_over_repetitions']:.1f} | "
                         f"{block['range_ms'][0]:.1f}-{block['range_ms'][1]:.1f} | {100*share:.0f}% |")
        lines += ['', f"Note: {timing['note']}", '']
    lines += ['Per test set, DSC (%):', '', '| configuration | '+' | '.join(TARGETS)+' |', '|---|'+'---|'*len(TARGETS)]
    for config in ('single30', 'single01', 'scz01', 't15'):
        lines.append(f'| {LABELS[config]} | '+' | '.join(f"{methods[config]['dsc'][t]:.2f}" for t in TARGETS)+' |')
    lines += ['', '## Definitions', '']+[f'- **{k}**: {v}' for k, v in protocol['definitions'].items()]
    lines += ['', f"Consequence fixed in advance: {rules['consequence']}"]
    lines += ['', '## Disclosures', '']+[f'- {d}' for d in protocol['disclosures']]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return str(REPORT/'report.md')
