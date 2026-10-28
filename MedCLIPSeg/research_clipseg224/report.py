"""Report of the CLIPSeg control from saved per-image files."""
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


def analyse(protocol, partial=False):
    statistics, seeds = protocol['statistics'], protocol['seeds']
    rng = np.random.default_rng(statistics['rng_seed'])
    draws, perms = statistics['bootstrap_draws'], statistics['permutation_draws']
    result = {}
    for group, spec in protocol['groups'].items():
        targets, ood = list(spec['targets']), list(spec['targets'])[1:]
        if not all(maybe(evaluation_dir(group, s, t)/'manifest.json').get('complete') for s in seeds for t in targets):
            if partial:
                continue
            raise RuntimeError(f'Evaluations of {group} are not complete')
        table, second, fraction = {}, {}, {}
        for s in seeds:
            for t in targets:
                for r in rows_of(evaluation_dir(group, s, t)/'cases.csv'):
                    table[(r['config'], t, s, r['name'])] = 100*float(r['DSC'])
                    second[(r['config'], t, s, r['name'])] = r['kind'] == 'in'
                    fraction[(t, r['name'])] = float(r['gt_fraction'])
        names = {t: sorted(n for (c, tt, s, n) in table if c == 'single' and tt == t and s == seeds[0]) for t in targets}
        for c in CONFIGS:
            for t in targets:
                for s in seeds:
                    if sorted(n for (cc, tt, ss, n) in table if cc == c and tt == t and ss == s) != names[t]:
                        raise RuntimeError(f'Cases do not match the cohort: {c}, {t}, seed {s}')
        methods = {}
        for c in CONFIGS:
            per = {t: [float(np.mean([table[(c, t, s, n)] for n in names[t]])) for s in seeds] for t in targets}
            ood_seed = [float(np.mean([per[t][i] for t in ood])) for i in range(len(seeds))]
            methods[c] = dict(dsc={t: dict(mean=float(np.mean(per[t])), sd=float(np.std(per[t], ddof=1)), per_seed=per[t]) for t in targets},
                              ood_dsc=dict(mean=float(np.mean(ood_seed)), sd=float(np.std(ood_seed, ddof=1)), per_seed=ood_seed))
        per_set = {t: [float(np.mean([table[('scz', t, s, n)]-table[('single', t, s, n)] for s in seeds])) for n in names[t]] for t in targets}
        per_seed = [float(np.mean([np.mean([table[('scz', t, s, n)]-table[('single', t, s, n)] for n in names[t]]) for t in ood])) for s in seeds]
        gain = dict(sets={t: stats(per_set[t], rng, draws, perms) for t in targets},
                    ood_macro=macro_stats({t: per_set[t] for t in ood}, rng, draws, perms), ood_macro_per_seed_pp=per_seed)
        share = float(np.mean([np.mean([second[('scz', t, s, n)] for s in seeds for n in names[t]]) for t in ood]))
        strata = []
        for low, high, label in ((0., .01, '<1%'), (.01, .03, '1-3%'), (.03, .059, '3-5.9%'), (.059, .15, '5.9-15%'), (.15, 1.01, '>=15%')):
            chosen = [(t, n) for t in ood for n in names[t] if low <= fraction[(t, n)] < high]
            if chosen:
                strata.append(dict(stratum=label, n=len(chosen),
                                   single=float(np.mean([table[('single', t, s, n)] for (t, n) in chosen for s in seeds])),
                                   scz=float(np.mean([table[('scz', t, s, n)] for (t, n) in chosen for s in seeds]))))
        low, high = gain['ood_macro']['ci95_pp']
        other_dir = lambda s, t: ROOT/'outputs_research'/COMPANION/'evaluations'/group/f'seed{s}'/t
        versus = None
        if all(maybe(other_dir(s, t)/'manifest.json').get('complete') for s in seeds for t in targets):
            other = {}
            for s in seeds:
                for t in targets:
                    for r in rows_of(other_dir(s, t)/'cases.csv'):
                        other[(r['config'], t, s, r['name'])] = 100*float(r['DSC'])
            if all((c, t, s, n) in other for c in CONFIGS for t in targets for s in seeds for n in names[t]):
                extra = lambda t, s, n: (table[('scz', t, s, n)]-table[('single', t, s, n)])-(other[('scz', t, s, n)]-other[('single', t, s, n)])
                per = {t: [float(np.mean([extra(t, s, n) for s in seeds])) for n in names[t]] for t in ood}
                their_single = [float(np.mean([np.mean([other[('single', t, s, n)] for n in names[t]]) for t in ood])) for s in seeds]
                their_scz = [float(np.mean([np.mean([other[('scz', t, s, n)] for n in names[t]]) for t in ood])) for s in seeds]
                versus = dict(gain_here_minus_gain_at_352=macro_stats(per, rng, draws, perms),
                              per_seed_pp=[float(np.mean([np.mean([extra(t, s, n) for n in names[t]]) for t in ood])) for s in seeds],
                              ood_dsc_352=dict(single=float(np.mean(their_single)), scz=float(np.mean(their_scz))))
        training = [read(training_dir(group, s)/'manifest.json') for s in seeds]
        result[group] = dict(targets=targets, methods=methods, scz_minus_single=gain, second_pass_share_ood=share, by_lesion_area_ood=strata,
                             verdicts=dict(transfers=bool(low > 0 and all(v > 0 for v in per_seed)), does_not_transfer=bool(low <= 0),
                                           larger_at_low_resolution=None if versus is None else bool(versus['gain_here_minus_gain_at_352']['ci95_pp'][0] > 0)),
                             versus_352=versus,
                             training_minutes_per_seed=[t['active_seconds']/60 for t in training],
                             final_training_loss=[t['history'][-1]['loss'] for t in training])
    return result


def ci(s):
    return f"{s['mean_pp']:+.2f} [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}], p={s['p']:.4f}"


def write_report(partial=False):
    protocol = load_protocol()
    analysis = analyse(protocol, partial)
    plan = read(OUTPUT/'plan.json')
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', dict(name=NAME, created_at=now(), plan_digest=plan['digest'], partial=partial,
                                           check=maybe(CONTROL/'check.json'), groups=analysis))
    lines = [f'# {NAME}: scale-canonical zoom on CLIPSeg at 224 x 224 (frozen CLIP encoders, fixed sentence, three seeds)', '',
             'Decoder fine-tuned on the Kvasir-SEG training split, last-epoch weights, no selection on validation or test data. '
             'OOD = equal-weight mean of the four OOD polyp sets.', '']
    if partial:
        lines += ['**Partial report.**', '']
    for group, block in analysis.items():
        targets, gain, verdicts = block['targets'], block['scz_minus_single'], block['verdicts']
        lines += ['| inference | '+' | '.join(targets)+' | OOD mean (SD over seeds) |', '|---|'+'---|'*(len(targets)+1)]
        for c, label in (('single', 'single pass'), ('scz', 'SCZ (locked rule)')):
            m = block['methods'][c]
            lines.append(f'| {label} | '+' | '.join(f"{m['dsc'][t]['mean']:.2f}" for t in targets)+f" | {m['ood_dsc']['mean']:.2f} ({m['ood_dsc']['sd']:.2f}) |")
        lines.append('| SCZ minus single pass (pp) | '+' | '.join(ci(gain['sets'][t]) for t in targets)+f" | {ci(gain['ood_macro'])} |")
        lines += ['', f"OOD gain per seed: {', '.join(f'{v:+.2f}' for v in gain['ood_macro_per_seed_pp'])}. "
                      f"Second pass on {100*block['second_pass_share_ood']:.0f}% of the OOD images.", '',
                  'OOD images pooled, by lesion area (DSC %, mean over seeds):', '', '| lesion area | n | single pass | SCZ |', '|---|---|---|---|']
        lines += [f"| {r['stratum']} | {r['n']} | {r['single']:.1f} | {r['scz']:.1f} |" for r in block['by_lesion_area_ood']]
        lines += ['', '## Verdicts by the rules fixed in advance', '',
                  f"- SCZ transfers to CLIPSeg (gain positive for every seed, interval excludes zero): **{verdicts['transfers']}**",
                  f"- SCZ does not transfer (interval includes zero or lies below it): **{verdicts['does_not_transfer']}**",
                  f"- The gain is larger at 224 x 224 than at 352 x 352: **{verdicts['larger_at_low_resolution']}**"
                  +('' if block['versus_352'] is None else f" (difference of the OOD gains: {ci(block['versus_352']['gain_here_minus_gain_at_352'])}; per seed "
                    f"{', '.join(f'{v:+.2f}' for v in block['versus_352']['per_seed_pp'])}; at 352 x 352: single pass {block['versus_352']['ood_dsc_352']['single']:.2f}, "
                    f"SCZ {block['versus_352']['ood_dsc_352']['scz']:.2f})"), '',
                  f"Training time per seed: {', '.join(f'{m:.0f} min' for m in block['training_minutes_per_seed'])}. "
                  f"Final training loss: {', '.join(f'{v:.4f}' for v in block['final_training_loss'])}.", '']
    lines += [f"Consequence fixed in advance: {protocol['reading_rules_fixed_in_advance']['consequence']}", '', '## Disclosures', '']
    lines += [f'- {d}' for d in protocol['disclosures']]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return str(REPORT/'report.md')
