"""Report of two further seeds of the Polyp-PVT reference from saved per-image files."""
import numpy as np

from research_polyp_pvt.report import load_cases, stats, macro_stats, mean_sd, ci, ms
from .common import *


def summarise(cases, targets, seeds):
    """cases[seed][target] -> name -> record."""
    ood = targets[1:]
    names = {t: sorted(cases[seeds[0]][t]) for t in targets}
    for s in seeds:
        for t in targets:
            if sorted(cases[s][t]) != names[t]:
                raise RuntimeError(f'Case list differs: {t}, seed {s}')
    block = dict(sets={})
    for t in targets:
        lesion = [n for n in names[t] if cases[seeds[0]][t][n]['gt_pixels'] > 0]
        normal = [n for n in names[t] if cases[seeds[0]][t][n]['gt_pixels'] == 0]
        block['sets'][t] = dict(
            n=len(names[t]), n_lesion_free=len(normal),
            DSC=mean_sd([np.mean([cases[s][t][n]['DSC'] for n in names[t]]) for s in seeds]),
            NSD=mean_sd([np.mean([cases[s][t][n]['NSD'] for n in names[t]]) for s in seeds]),
            missed_lesions_per_seed=[int(sum(cases[s][t][n]['pred_pixels'] == 0 for n in lesion)) for s in seeds],
            lesion_free_correct_per_seed=[int(sum(cases[s][t][n]['pred_pixels'] == 0 for n in normal)) for s in seeds])
    for key in ('DSC', 'NSD'):
        block['OOD_'+key] = mean_sd([np.mean([np.mean([cases[s][t][n][key] for n in names[t]]) for t in ood]) for s in seeds])
    block['OOD_missed_lesions_per_seed'] = [int(sum(block['sets'][t]['missed_lesions_per_seed'][i] for t in ood)) for i in range(len(seeds))]
    return block, names


def analyse(protocol, partial=False):
    statistics, seeds = protocol['statistics'], protocol['seeds']
    rng = np.random.default_rng(statistics['rng_seed'])
    draws, perms = statistics['bootstrap_draws'], statistics['permutation_draws']
    result = {}
    for group, spec in protocol['groups'].items():
        targets = list(spec['targets'])
        ready = all(maybe(evaluation_dir(group, s, t)/'manifest.json').get('complete') for s in seeds for t in targets)
        if not ready:
            if partial:
                continue
            raise RuntimeError(f'Evaluations of {group} are not complete')
        mine = {s: {t: load_cases(evaluation_dir(group, s, t)/'cases.csv') for t in targets} for s in seeds}
        summary, names = summarise(mine, targets, seeds)
        training = [read(training_dir(group, s)/'manifest.json') for s in seeds]
        block = dict(targets=targets, pvt=summary, training_hours_per_seed=[t['active_seconds']/3600 for t in training],
                     final_training_loss=[t['history'][-1]['loss'] for t in training],
                     last_ten_epoch_loss_range=[[min(h['loss'] for h in t['history'][-10:]), max(h['loss'] for h in t['history'][-10:])]
                                                for t in training],
                     ours={}, contrasts={})
        for model, info in protocol['comparison'].get(group, {}).items():
            ours = {s: {t: load_cases(*accepted_cases(protocol, group, model, s, t)) for t in targets} for s in seeds}
            block['ours'][model], other_names = summarise(ours, targets, seeds)
            if other_names != names:
                raise RuntimeError(f'Case lists of {model} differ from this evaluation')
            diff = {t: np.array([np.mean([mine[s][t][n]['DSC']-ours[s][t][n]['DSC'] for s in seeds]) for n in names[t]]) for t in targets}
            block['contrasts'][model] = dict(
                sets={t: stats(diff[t], rng, draws, perms) for t in targets},
                ood_macro=macro_stats({t: diff[t] for t in targets[1:]}, rng, draws, perms),
                ood_macro_per_seed_pp=[float(np.mean([np.mean([mine[s][t][n]['DSC']-ours[s][t][n]['DSC'] for n in names[t]])
                                                      for t in targets[1:]])) for s in seeds])
        result[group] = block
    return result


def write_report(partial=False):
    protocol = load_protocol()
    analysis = analyse(protocol, partial)
    plan = read(OUTPUT/'plan.json')
    held, seeds = protocol['held_out_target'], protocol['seeds']
    for group, block in analysis.items():
        cases = {s: load_cases(evaluation_dir(group, s, held)/'cases.csv') for s in seeds
                 if maybe(evaluation_dir(group, s, held)/'manifest.json').get('complete')}
        if len(cases) == len(seeds):
            block['held_out'] = dict(target=held, n=len(cases[seeds[0]]),
                                     DSC=mean_sd([np.mean([r['DSC'] for r in cases[s].values()]) for s in seeds]),
                                     NSD=mean_sd([np.mean([r['NSD'] for r in cases[s].values()]) for s in seeds]))
        elif not partial:
            raise RuntimeError('Evaluations of the held-out set are not complete')
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', dict(name=NAME, created_at=now(), plan_digest=plan['digest'], partial=partial, groups=analysis))
    lines = [f'# {NAME}: Polyp-PVT reference, training seeds {seeds} (full Kvasir-SEG training set)', '',
             'Released recipe unchanged, last-epoch weights, no selection on validation or test data. '
             'OOD = equal-weight mean of the four OOD polyp sets.', '']
    if partial:
        lines += ['**Partial report.**', '']
    for group, block in analysis.items():
        targets = block['targets']
        lines += ['| seed | '+' | '.join(targets)+' | OOD mean | '+held+' |', '|---|'+'---|'*(len(targets)+2)]
        for i, seed in enumerate(seeds):
            cells = [f"{block['pvt']['sets'][t]['DSC']['per_seed'][i]:.2f}" for t in targets]
            cells.append(f"{block['pvt']['OOD_DSC']['per_seed'][i]:.2f}")
            cells.append(f"{block['held_out']['DSC']['per_seed'][i]:.2f}" if 'held_out' in block else '')
            lines.append(f'| {seed} | '+' | '.join(cells)+' |')
        lines += ['', f"Missed lesions on the four OOD sets (per seed): {block['pvt']['OOD_missed_lesions_per_seed']}. "
                      f"Training time per seed: {', '.join(f'{h:.2f} h' for h in block['training_hours_per_seed'])}. "
                      f"Final training loss: {', '.join(f'{v:.4f}' for v in block['final_training_loss'])}.", '']
    lines += ['## Reading rule fixed in advance', '', protocol['reading_rule_fixed_in_advance'], '', '## Disclosures', '']
    lines += [f'- {d}' for d in protocol['disclosures']]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return str(REPORT/'report.md')
