"""Report of the test-time text analysis, the one-pass run and the timing, from saved per-image files."""
import csv

import numpy as np

from .common import *

LABELS = dict(fixed='training sentence (accepted single pass)', empty='empty text', wrong_class='class sentence of the other modality',
              paraphrase_a='other wording A', paraphrase_b='other wording B', single01='training sentence, one stochastic pass')


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
    table, texts = {}, {}
    for run in protocol['runs']:
        seed = seed_of(run)
        for target in targets:
            folder = OUTPUT/'evaluations'/run/target
            if not maybe(folder/'manifest.json').get('complete'):
                raise RuntimeError(f'Evaluation not complete: {run}/{target}')
            for r in rows_of(folder/'cases.csv'):
                if r['config'] not in CONFIGS:
                    raise RuntimeError('Unexpected configuration: '+r['config'])
                table[(r['config'], target, seed, r['name'])] = 100*float(r['DSC'])
            used = read(folder/'texts.json')
            if texts and used != texts:
                raise RuntimeError('The text conditions differ between evaluations')
            texts = used
    names = {t: sorted({n for (c, tt, s, n) in table if c == 'fixed' and tt == t}) for t in targets}
    for config in CONFIGS:
        for t in targets:
            for s in seeds:
                if sorted(n for (c, tt, ss, n) in table if c == config and tt == t and ss == s) != names[t]:
                    raise RuntimeError(f'Cases do not match the cohort: {config}, {t}, seed {s}')
    methods = {}
    for config in CONFIGS:
        per_seed = {t: [float(np.mean([table[(config, t, s, n)] for n in names[t]])) for s in seeds] for t in targets}
        ood_seed = [float(np.mean([per_seed[t][i] for t in ood])) for i in range(len(seeds))]
        block = dict(dsc={t: float(np.mean(per_seed[t])) for t in targets}, dsc_per_seed=per_seed,
                     ood_dsc=float(np.mean(ood_seed)), ood_dsc_sd=float(np.std(ood_seed, ddof=1)), ood_dsc_per_seed=ood_seed)
        if config != 'fixed':
            per_set = {t: [float(np.mean([table[(config, t, s, n)]-table[('fixed', t, s, n)] for s in seeds])) for n in names[t]]
                       for t in targets}
            block['minus_fixed'] = dict(
                sets={t: stats(per_set[t], rng, draws, perms) for t in targets},
                ood_macro=macro_stats({t: per_set[t] for t in ood}, rng, draws, perms),
                ood_macro_per_seed_pp=[float(np.mean([np.mean([table[(config, t, s, n)]-table[('fixed', t, s, n)] for n in names[t]])
                                                      for t in ood])) for s in seeds])
        methods[config] = block
    macro = lambda config: methods[config]['minus_fixed']['ood_macro']
    excludes_zero = lambda s: s['ci95_pp'][0] > 0 or s['ci95_pp'][1] < 0
    verdicts = dict(
        text_content_matters=bool(macro('wrong_class')['mean_pp'] < 0 and macro('wrong_class')['ci95_pp'][1] < 0),
        acts_as_constant=bool(all(abs(macro(c)['mean_pp']) < .5 and not excludes_zero(macro(c)) for c in ('empty', 'wrong_class'))),
        wording_sensitive=bool(any(excludes_zero(macro(c)) for c in ('paraphrase_a', 'paraphrase_b'))))
    return dict(texts=texts, methods=methods, verdicts=verdicts, timing=maybe(OUTPUT/'timing.json'))


def ci(s):
    return f"{s['mean_pp']:+.2f} [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}], p={s['p']:.4f}"


def write_report():
    protocol = load_protocol()
    analysis = analyse(protocol)
    plan = read(OUTPUT/'evaluation_plan.json')
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', dict(name=NAME, created_at=now(), plan_digest=plan['digest'], **analysis))
    methods, verdicts, timing = analysis['methods'], analysis['verdicts'], analysis['timing']
    strings = analysis['texts'].get('texts', {})
    lines = [f'# {NAME}: text at test time, one stochastic pass, and timing (detail-branch model, seeds 666-668)', '',
             'Nothing was trained. Single pass of the whole image, without SCZ. OOD = equal-weight mean of the four OOD polyp sets.', '',
             '| condition | text | Kvasir DSC | OOD DSC (SD over seeds) | minus the training sentence, OOD [95% CI], p | per seed |',
             '|---|---|---|---|---|---|']
    for config in CONFIGS:
        block = methods[config]
        text = strings.get('fixed' if config == 'single01' else config, '')
        delta = ci(block['minus_fixed']['ood_macro']) if 'minus_fixed' in block else ''
        seeds = ', '.join(f'{v:+.2f}' for v in block['minus_fixed']['ood_macro_per_seed_pp']) if 'minus_fixed' in block else ''
        lines.append(f"| {LABELS[config]} | `{text}` | {block['dsc'][TARGETS[0]]:.2f} | {block['ood_dsc']:.2f} ({block['ood_dsc_sd']:.2f}) | {delta} | {seeds} |")
    lines += ['', f"Empty text: {analysis['texts'].get('empty_text', 'n/a')}.", '', 'Per test set, DSC (%):', '',
              '| condition | '+' | '.join(TARGETS)+' |', '|---|'+'---|'*len(TARGETS)]
    for config in CONFIGS:
        lines.append(f'| {LABELS[config]} | '+' | '.join(f"{methods[config]['dsc'][t]:.2f}" for t in TARGETS)+' |')
    lines += ['', 'Difference to the training sentence per test set (pp):', '', '| condition | '+' | '.join(TARGETS)+' |', '|---|'+'---|'*len(TARGETS)]
    for config in CONFIGS[1:]:
        lines.append(f'| {LABELS[config]} | '+' | '.join(ci(methods[config]['minus_fixed']['sets'][t]) for t in TARGETS)+' |')
    lines += ['', '## Verdicts by the rules fixed in advance', '',
              f"- The content of the sentence matters (other-modality sentence lowers the OOD mean): **{verdicts['text_content_matters']}**",
              f"- The sentence acts as a constant input (empty and other-modality text both within 0.5 pp): **{verdicts['acts_as_constant']}**",
              f"- Sensitive to the wording (at least one other wording changes the OOD mean): **{verdicts['wording_sensitive']}**", '']
    if timing:
        lines += ['## Time of one inference run', '', f"{timing['model']} on {timing['dataset']}, {timing['device']}, batch size 1. {timing['note']}.", '',
                  '| stochastic passes | mean (ms) | SD (ms) | median (ms) | images |', '|---|---|---|---|---|']
        lines += [f"| {k} | {v['mean_ms']:.0f} | {v['sd_ms']:.0f} | {v['median_ms']:.0f} | {v['n']} |" for k, v in timing['passes'].items()]
        lines.append('')
    lines += ['## Definitions', '']+[f'- **{k}**: {v}' for k, v in protocol['definitions'].items()]
    lines += ['', f"Consequence fixed in advance: {protocol['reading_rules_fixed_in_advance']['consequence']}"]
    lines += ['', '## Disclosures', '']+[f'- {d}' for d in protocol['disclosures']]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return str(REPORT/'report.md')
