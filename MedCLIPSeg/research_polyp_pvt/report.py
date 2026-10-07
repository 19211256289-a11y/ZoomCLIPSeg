"""Report of the Polyp-PVT reference from saved per-image files. The rows of our models are the accepted results of the
same three seeds; nothing of ours is recomputed."""
import csv

import numpy as np

from .common import *

MODELS = ('pvt', 'baseline', 'full')


def load_cases(path, config=None):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if config is not None:
        rows = [r for r in rows if r['config'] == config]
    result = {r['name']: dict(DSC=100*float(r['DSC']), NSD=100*float(r['NSD']), gt_fraction=float(r['gt_fraction']),
                              gt_pixels=int(r['gt_pixels']), pred_pixels=int(r['pred_pixels'])) for r in rows}
    if not result or len(result) != len(rows):
        raise RuntimeError(f'Empty or duplicated cases: {path}')
    return result


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


def mean_sd(values):
    return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)), per_seed=[float(v) for v in values])


def analyse(protocol):
    statistics, seeds, threshold = protocol['statistics'], protocol['seeds'], protocol['small_target_threshold']
    rng = np.random.default_rng(statistics['rng_seed'])
    draws, perms = statistics['bootstrap_draws'], statistics['permutation_draws']
    targets = TARGETS+(HELD_OUT,)
    cases = {m: {t: {} for t in targets} for m in MODELS}
    for seed in seeds:
        for target in targets:
            cases['pvt'][target][seed] = load_cases(evaluation_dir(seed, target)/'cases.csv')
            for model in ('baseline', 'full'):
                cases[model][target][seed] = load_cases(*accepted_cases(protocol, model, seed, target))
    names = {t: sorted(cases['pvt'][t][seeds[0]]) for t in targets}
    for model in MODELS:
        for target in targets:
            for seed in seeds:
                if sorted(cases[model][target][seed]) != names[target]:
                    raise RuntimeError(f'Case list differs: {model}, {target}, seed {seed}')
    small = {t: [n for n in names[t] if 0 < cases['pvt'][t][seeds[0]][n]['gt_pixels'] and cases['pvt'][t][seeds[0]][n]['gt_fraction'] <= threshold]
             for t in targets}
    positive = {t: [n for n in names[t] if cases['pvt'][t][seeds[0]][n]['gt_pixels'] > 0] for t in targets}

    def per_image(model, target, key='DSC'):
        """Seed-averaged value of every image."""
        return np.array([np.mean([cases[model][target][s][n][key] for s in seeds]) for n in names[target]])

    summary = {}
    for model in MODELS:
        block = dict(sets={})
        for target in targets:
            c = cases[model][target]
            block['sets'][target] = dict(
                n=len(names[target]),
                DSC=mean_sd([np.mean([c[s][n]['DSC'] for n in names[target]]) for s in seeds]),
                NSD=mean_sd([np.mean([c[s][n]['NSD'] for n in names[target]]) for s in seeds]),
                n_small=len(small[target]),
                small_DSC=float(np.mean([c[s][n]['DSC'] for s in seeds for n in small[target]])) if small[target] else None,
                empty_predictions_per_seed=[int(sum(c[s][n]['pred_pixels'] == 0 for n in positive[target])) for s in seeds],
                DSC_below_10_per_seed=[int(sum(c[s][n]['DSC'] < 10 for n in positive[target])) for s in seeds])
        for key in ('DSC', 'NSD'):
            block['OOD_'+key] = mean_sd([np.mean([np.mean([cases[model][t][s][n][key] for n in names[t]]) for t in OOD]) for s in seeds])
        pooled = [cases[model][t][s][n]['DSC'] for t in OOD for s in seeds for n in small[t]]
        block['OOD_small_DSC'] = float(np.mean(pooled))
        block['OOD_n_small'] = int(sum(len(small[t]) for t in OOD))
        summary[model] = block
    contrasts = {}
    for other in ('baseline', 'full'):
        diff = {t: per_image('pvt', t)-per_image(other, t) for t in targets}
        index = {t: {n: i for i, n in enumerate(names[t])} for t in targets}
        small_ood = np.concatenate([[diff[t][index[t][n]] for n in small[t]] for t in OOD])
        contrasts[other] = dict(
            sets={t: stats(diff[t], rng, draws, perms) for t in targets},
            ood_macro=macro_stats({t: diff[t] for t in OOD}, rng, draws, perms),
            ood_macro_per_seed_pp=[float(np.mean([np.mean([cases['pvt'][t][s][n]['DSC']-cases[other][t][s][n]['DSC'] for n in names[t]])
                                                  for t in OOD])) for s in seeds],
            ood_small_pooled=stats(small_ood, rng, draws, perms),
            held_out_small=stats([diff[HELD_OUT][index[HELD_OUT][n]] for n in small[HELD_OUT]], rng, draws, perms) if small[HELD_OUT] else None)
    return dict(summary=summary, contrasts=contrasts)


def ci(s):
    return f"{s['mean_pp']:+.2f} [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}], p={s['p']:.4f}"


def ms(v):
    return f"{v['mean']:.2f} +/- {v['sd']:.2f}"


def write_report():
    protocol = load_protocol()
    analysis = analyse(protocol)
    plan, check = read(OUTPUT/'plan.json'), maybe(CONTROL/'check.json')
    seeds, targets = protocol['seeds'], TARGETS+(HELD_OUT,)
    training = [read(training_dir(s)/'manifest.json') for s in seeds]
    timing = [read(evaluation_dir(s, protocol['timing']['dataset'])/'timing.json') for s in seeds]
    cost = dict(parameters=check.get('parameters'), trainable_parameters=check.get('trainable_parameters'),
                peak_training_memory_gb=check.get('peak_gpu_memory_gb'),
                training_hours_per_seed=[t['active_seconds']/3600 for t in training],
                final_training_loss=[t['history'][-1]['loss'] for t in training],
                inference_ms_per_image=float(np.mean([t['mean_ms'] for t in timing])), device=timing[0]['device'])
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', dict(name=NAME, created_at=now(), plan_digest=plan['digest'], cost=cost, **analysis))
    labels = dict(pvt='Polyp-PVT (this run)', **{k: v['label'] for k, v in protocol['comparison']['ours'].items()})
    summary, contrasts = analysis['summary'], analysis['contrasts']
    lines = [f'# {NAME}: Polyp-PVT trained on the 800 Kvasir-SEG training images', '',
             f'Three seeds ({", ".join(map(str, seeds))}), last-epoch weights, no selection on validation or test data. '
             'Rows of our models: accepted results of the same three seeds. OOD = equal-weight mean of the four OOD sets. '
             f'{HELD_OUT} is the held-out set.', '']
    for key in ('DSC', 'NSD'):
        lines += [f'## {key} (%), mean +/- SD over seeds', '', '| model | '+' | '.join(TARGETS)+f' | OOD mean | {HELD_OUT} |', '|---|'+'---|'*(len(TARGETS)+2)]
        for model in MODELS:
            block = summary[model]
            lines.append(f'| {labels[model]} | '+' | '.join(ms(block['sets'][t][key]) for t in TARGETS)
                         + f" | {ms(block['OOD_'+key])} | {ms(block['sets'][HELD_OUT][key])} |")
        lines.append('')
    lines += ['## Polyp-PVT minus our models, DSC in percentage points [95% CI], p', '',
              '| against | '+' | '.join(targets)+' | OOD mean | OOD mean per seed |', '|---|'+'---|'*(len(targets)+2)]
    for other, block in contrasts.items():
        lines.append(f'| {labels[other]} | '+' | '.join(ci(block['sets'][t]) for t in targets)+f" | {ci(block['ood_macro'])} | "
                     + ', '.join(f'{v:+.2f}' for v in block['ood_macro_per_seed_pp'])+' |')
    lines += ['', f"## Small lesions (area fraction at most {protocol['small_target_threshold']:.4f}), DSC (%)", '',
              f"| model | OOD pooled (n={summary['pvt']['OOD_n_small']}) | {HELD_OUT} (n={summary['pvt']['sets'][HELD_OUT]['n_small']}) |", '|---|---|---|']
    for model in MODELS:
        held = summary[model]['sets'][HELD_OUT]['small_DSC']
        lines.append(f"| {labels[model]} | {summary[model]['OOD_small_DSC']:.2f} | {'' if held is None else f'{held:.2f}'} |")
    lines.append('')
    for other, block in contrasts.items():
        held = block['held_out_small']
        lines.append(f"- Polyp-PVT minus {labels[other]}: OOD pooled {ci(block['ood_small_pooled'])}"
                     + (f"; {HELD_OUT} {ci(held)}" if held else ''))
    lines += ['', '## Images without any predicted lesion / with DSC below 10% (lesion images, per seed)', '',
              '| model | '+' | '.join(targets)+' |', '|---|'+'---|'*len(targets)]
    for model in MODELS:
        lines.append(f'| {labels[model]} | '+' | '.join(
            '/'.join(map(str, summary[model]['sets'][t]['empty_predictions_per_seed']))+' ; '
            + '/'.join(map(str, summary[model]['sets'][t]['DSC_below_10_per_seed'])) for t in targets)+' |')
    lines += ['', '## Cost of Polyp-PVT', '']
    if cost['parameters']:
        lines.append(f"- Parameters: {cost['parameters']/1e6:.2f} M, all trainable ({cost['trainable_parameters']/1e6:.2f} M).")
        lines.append(f"- Peak GPU memory in training (batch 16, largest size 448): {cost['peak_training_memory_gb']:.1f} GB.")
    lines += [f"- Training time per seed: {', '.join(f'{h:.2f} h' for h in cost['training_hours_per_seed'])} on {cost['device']}.",
              f"- Final training loss per seed: {', '.join(f'{v:.4f}' for v in cost['final_training_loss'])}.",
              f"- Inference: {cost['inference_ms_per_image']:.1f} ms per image ({protocol['timing']['note']}; "
              f"{protocol['timing']['dataset']}, after {protocol['timing']['warmup_images']} warm-up images).",
              '', '## Differences from the released training and test scripts', '']
    lines += [f'- {d}' for d in protocol['deviations_from_the_authors']]
    lines += ['', '## Disclosures', '']+[f'- {d}' for d in protocol['disclosures']]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return str(REPORT/'report.md')
