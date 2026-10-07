"""Step 0: image-level analysis of the twelve accepted source-Val evaluations. No model, no GPU, no Test data."""
import csv

from .common import MODES, SEEDS, SMALL, STEP0, RUNS, load_protocol, accepted_runs, val_cases_path, split_run, write_json, now, sha256
from .stats import load_cases, arm_summary, is_small
from .report import analyse, section


def main():
    protocol = load_protocol()
    accepted = accepted_runs(protocol)
    cases = {mode: {} for mode in MODES}
    sources = {}
    for run in RUNS:
        mode, seed = split_run(run)
        path = val_cases_path(run)
        rows = load_cases(path)
        summary, expected = arm_summary(rows, SMALL), accepted[run]['summary']
        for key in ('DSC_percent', 'small_DSC_percent', 'NSD_percent'):
            if abs(summary[key]-expected[key]) > 1e-6:
                raise ValueError(f'{run}: {key} recomputed from cases differs from the accepted audit')
        if (summary['n'], summary['n_small']) != (100, 25):
            raise ValueError(f'{run}: unexpected cohort size')
        cases[mode][seed] = rows
        sources[run] = dict(path=str(path), sha256=sha256(path))
    result = analyse(cases, protocol)
    STEP0.mkdir(parents=True, exist_ok=True)
    write_json(STEP0/'results.json', dict(scope='Kvasir source Val, n=100, 25 small; saved accepted predictions only',
                                          sources=sources, result=result, generated_at=now()))
    reference = {r['name']: r for r in cases['baseline'][SEEDS[0]]}
    tables = {(m, s): {r['name']: r for r in cases[m][s]} for m in MODES for s in SEEDS}
    with (STEP0/'per_image.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ['name', 'gt_fraction', 'small'] + [f'DSC_{m}_seed{s}' for m in MODES for s in SEEDS] + \
                 [f'mean_diff_pp_{lhs}_minus_{rhs}' for lhs, rhs in protocol['contrasts']]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in sorted(reference, key=lambda n: reference[n]['gt_fraction']):
            row = dict(name=name, gt_fraction=reference[name]['gt_fraction'], small=int(is_small(reference[name], SMALL)))
            for m in MODES:
                for s in SEEDS:
                    row[f'DSC_{m}_seed{s}'] = tables[m, s][name]['DSC']
            for lhs, rhs in protocol['contrasts']:
                row[f'mean_diff_pp_{lhs}_minus_{rhs}'] = round(100*sum(tables[lhs, s][name]['DSC']-tables[rhs, s][name]['DSC'] for s in SEEDS)/len(SEEDS), 4)
            writer.writerow(row)
    lines = ['# Step 0: image-level analysis of the accepted Val predictions', '',
             'Source Kvasir Val (100 images, 25 small targets), four arms x three seeds, epoch 100, generic prompt.',
             'Recomputed means match the three acceptance audits. This is the repeatedly used development set, not confirmation data.', '',
             *section('Kvasir Val', result)]
    (STEP0/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Step 0 report:', STEP0/'report.md', flush=True)
    return STEP0/'report.md'


if __name__ == '__main__':
    main()
