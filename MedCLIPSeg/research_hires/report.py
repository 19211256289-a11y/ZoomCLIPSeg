"""Val tables, paired contrasts and the pre-specified gate for the native-resolution pilot."""
from research_detail_test.stats import load_cases
from research_detail_ext.report import analyse, section, f
from .common import OUTPUT, REPORT, load_suite, maybe, read, write_json, now


def evaluate_gate(result, suite):
    gate, checks = suite['gate'], {}
    a = result['contrasts']['detail448 - detail224']
    rule = gate['detail448_vs_detail224']
    small, nsd, overall = a['DSC']['small'], a['NSD']['all'], a['DSC']['all']
    checks['448 vs 224 detail: small-target DSC gain'] = dict(value=small['mean_pp'], need=f">= +{rule['small_dsc_gain_min_pp']}",
                                                              passed=small['mean_pp'] >= rule['small_dsc_gain_min_pp'])
    checks['448 vs 224 detail: NSD gain'] = dict(value=nsd['mean_pp'], need=f">= +{rule['nsd_gain_min_pp']}",
                                                 passed=nsd['mean_pp'] >= rule['nsd_gain_min_pp'])
    checks['448 vs 224 detail: overall DSC'] = dict(value=overall['mean_pp'], need=f">= -{rule['overall_dsc_max_drop_pp']}",
                                                    passed=overall['mean_pp'] >= -rule['overall_dsc_max_drop_pp'])
    checks['448 vs 224 detail: small gain positive in both seeds'] = dict(
        value=min(small['per_seed_mean_pp']), need='> 0', passed=all(v > 0 for v in small['per_seed_mean_pp']))
    b = result['contrasts']['detail448 - sup448']['DSC']['small']
    rule = gate['detail448_vs_sup448']
    checks['448 detail vs 448 supervision only: small-target DSC gain'] = dict(
        value=b['mean_pp'], need=f">= +{rule['small_dsc_gain_min_pp']}", passed=b['mean_pp'] >= rule['small_dsc_gain_min_pp'])
    checks['448 detail vs 448 supervision only: positive in both seeds'] = dict(
        value=min(b['per_seed_mean_pp']), need='> 0', passed=all(v > 0 for v in b['per_seed_mean_pp']))
    return dict(checks=checks, passed=all(c['passed'] for c in checks.values()))


def write_report():
    suite = load_suite()
    modes, seeds = suite['modes'], suite['seeds']
    cases, summaries = {mode: {} for mode in modes}, {}
    for mode in modes:
        for seed in seeds:
            folder = OUTPUT/'evaluations'/f'{mode}_seed{seed}'/'Val'
            if not maybe(folder/'manifest.json').get('complete'):
                raise ValueError(f'Incomplete evaluation: {mode}_seed{seed}')
            cases[mode][seed] = load_cases(folder/'cases.csv')
            summaries[f'{mode}_seed{seed}'] = read(folder/'summary.json')
    result = analyse(cases, modes, seeds, suite, suite['small_target_threshold'])
    gate = evaluate_gate(result, suite)
    reproduction = {}
    for run, reference in suite['reference_30_epoch_results'].items():
        if run == 'note':
            continue
        reproduction[run] = dict(reference=reference['DSC_percent'], now=summaries[run]['DSC_percent'],
                                 difference_pp=summaries[run]['DSC_percent']-reference['DSC_percent'])
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', dict(suite=suite, gate=gate, result=result, reproduction=reproduction,
                                           summaries=summaries, generated_at=now()))
    mark = lambda ok: 'PASS' if ok else 'FAIL'
    lines = [f"# {suite['name']}: native-resolution pilot", '',
             'Kvasir source Val (100 images, 25 small), seeds 666 and 667, 30 epochs, fixed sentence, MC30. No Test data was used.',
             'Two seeds and one development set: this is a screening result, not confirmation.', '',
             f"## Gate: **{mark(gate['passed'])}**", '', '| Criterion | Value pp | Required | Result |', '|---|---:|---|---|']
    lines += [f"| {name} | {f(c['value'], 2, True)} | {c['need']} | {mark(c['passed'])} |" for name, c in gate['checks'].items()]
    lines += ['', f"Next step if passed: {suite['gate']['if_pass']}. If failed: {suite['gate']['if_fail']}.", '',
              '## Reproduction of the earlier 30-epoch pilots', '', '| Run | Earlier DSC % | This suite DSC % | Difference pp |', '|---|---:|---:|---:|']
    lines += [f"| {run} | {f(r['reference'], 4)} | {f(r['now'], 4)} | {f(r['difference_pp'], 4, True)} |" for run, r in reproduction.items()]
    lines += ['', '## Results', '', *section('Kvasir Val', result, modes)]
    (REPORT/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return REPORT/'report.md'
