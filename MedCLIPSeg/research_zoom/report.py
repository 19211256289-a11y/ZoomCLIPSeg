"""Aggregation, pre-specified selection and gate. numpy only."""
import csv

import numpy as np

from .common import OUTPUT, REPORT, RUNS, load_protocol, write_json, now, maybe


def load(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        for k in ('DSC', 'NSD', 'gt_fraction', 'window'):
            r[k] = float(r[k])
        for k in ('gt_pixels', 'pred_pixels', 'fallback'):
            r[k] = int(float(r[k]))
    return rows


def analyse(cases, protocol):
    """cases: {(run, condition): rows}. Returns the result dictionary."""
    conditions = [c['name'] for c in protocol['conditions']]
    configs = [c['name'] for c in protocol['configs']]
    candidates = [c['name'] for c in protocol['configs'] if not c['oracle']]
    table, detail = {}, {}
    for condition in conditions:
        table[condition], detail[condition] = {}, {}
        for config in ['single']+configs:
            dsc, nsd, info = [], [], dict(zoom_in=[], zoom_out=[], fallback=[], improved=[], worsened=[], failures=[])
            for run in RUNS:
                rows = cases[run, condition]
                base = {r['name']: r for r in rows if r['config'] == 'single'}
                mine = [r for r in rows if r['config'] == config]
                if len(mine) != len(base):
                    raise ValueError('Incomplete cases')
                dsc.append(100*float(np.mean([r['DSC'] for r in mine])))
                nsd.append(100*float(np.mean([r['NSD'] for r in mine])))
                diff = [100*(r['DSC']-base[r['name']]['DSC']) for r in mine]
                info['zoom_in'].append(sum(r['kind'] == 'in' for r in mine))
                info['zoom_out'].append(sum(r['kind'] == 'out' for r in mine))
                info['fallback'].append(sum(r['fallback'] for r in mine))
                info['improved'].append(sum(d > 1 for d in diff))
                info['worsened'].append(sum(d < -1 for d in diff))
                info['failures'].append(sum(r['gt_pixels'] > 0 and r['DSC'] < .5 for r in mine))
            table[condition][config] = dict(DSC_per_seed=dsc, DSC=float(np.mean(dsc)), NSD=float(np.mean(nsd)))
            detail[condition][config] = {k: float(np.mean(v)) for k, v in info.items()}
        rows = [r for r in cases[RUNS[0], condition] if r['config'] == 'single']
        positive = [r['gt_fraction'] for r in rows if r['gt_pixels'] > 0]
        detail[condition]['cohort'] = dict(n=len(rows), n_positive=len(positive),
                                           median_lesion_fraction=float(np.median(positive)) if positive else None,
                                           n_below_1_percent=sum(f < .01 for f in positive))
    score = {c: float(np.mean([table[k][c]['DSC'] for k in conditions])) for c in candidates}
    selected = max(candidates, key=lambda c: score[c])  # max keeps the first listed on ties
    gate = protocol['gate']
    delta = lambda k: np.array(table[k][selected]['DSC_per_seed'])-np.array(table[k]['single']['DSC_per_seed'])
    shrink = np.mean([delta(k) for k in gate['shrink_conditions']], axis=0)
    smallest = min(protocol['conditions'], key=lambda c: c['scale'])['name']
    drop = table['s100']['single']['DSC']-table[smallest]['single']['DSC']
    checks = dict(
        p0=bool(drop >= gate['p0_proxy_valid_single_pass_drop_s025_vs_s100_at_least_pp']),
        g1=bool(shrink.mean() >= gate['g1_mean_shrink_dsc_gain_at_least_pp']),
        g2=bool(delta('s100').mean() >= gate['g2_unshifted_dsc_change_at_least_pp']),
        g3=bool((shrink > 0).all()))
    verdict = 'proxy_invalid' if not checks['p0'] else 'pass' if all(checks.values()) else 'fail'
    return dict(table=table, detail=detail, selection_score=score, selected=selected, checks=checks, verdict=verdict,
                single_pass_drop_pp=float(drop), shrink_gain_per_seed_pp=[float(v) for v in shrink],
                shrink_gain_pp=float(shrink.mean()), unshifted_change_pp=float(delta('s100').mean()),
                per_condition_gain_pp={k: float(delta(k).mean()) for k in conditions})


def write_report():
    protocol = load_protocol()
    cases = {}
    for run in RUNS:
        for condition in protocol['conditions']:
            folder = OUTPUT/'evaluations'/run/condition['name']
            if not maybe(folder/'manifest.json').get('complete'):
                raise ValueError('Evaluation not complete: '+str(folder))
            cases[run, condition['name']] = load(folder/'cases.csv')
    result = analyse(cases, protocol)
    result.update(name=protocol['name'], created_at=now())
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', result)
    conditions = [c['name'] for c in protocol['conditions']]
    configs = ['single']+[c['name'] for c in protocol['configs']]
    lines = ['# Pilot C: scale-canonical two-pass inference (Kvasir Val, synthetic scale shift)', '',
             'Evaluation only. Three-seed mean DSC (%) of the detail arm. No Test or OOD image was read.', '',
             f"**Verdict: {result['verdict']}** | selected config: `{result['selected']}`", '',
             '## Cohorts', '', '| condition | n | positive | median lesion fraction | lesions <1% |', '|---|---|---|---|---|']
    for k in conditions:
        c = result['detail'][k]['cohort']
        lines.append(f"| {k} | {c['n']} | {c['n_positive']} | {100*c['median_lesion_fraction']:.2f}% | {c['n_below_1_percent']} |")
    lines += ['', '## DSC by condition and config', '', '| config | '+' | '.join(conditions)+' | mean |', '|---|'+'---|'*(len(conditions)+1)]
    for c in configs:
        values = [result['table'][k][c]['DSC'] for k in conditions]
        lines.append(f'| {c} | '+' | '.join(f'{v:.2f}' for v in values)+f' | {np.mean(values):.2f} |')
    lines += ['', f"## Selected config `{result['selected']}` versus single-pass", '',
              '| condition | gain (pp) | zoom-in | zoom-out | fallback | improved >1pp | worsened >1pp | failures single -> two-pass |',
              '|---|---|---|---|---|---|---|---|']
    for k in conditions:
        d, s = result['detail'][k][result['selected']], result['detail'][k]['single']
        lines.append(f"| {k} | {result['per_condition_gain_pp'][k]:+.2f} | {d['zoom_in']:.1f} | {d['zoom_out']:.1f} | "
                     f"{d['fallback']:.1f} | {d['improved']:.1f} | {d['worsened']:.1f} | {s['failures']:.1f} -> {d['failures']:.1f} |")
    lines += ['', '(counts are means over the three seeds; failures are positive images with DSC < 50%)', '', '## Gate', '',
              f"- p0 proxy valid (single-pass drop at the smallest scale {result['single_pass_drop_pp']:.2f} pp, need >= 5.0): {result['checks']['p0']}",
              f"- g1 mean shrink gain {result['shrink_gain_pp']:+.2f} pp (need >= +2.0): {result['checks']['g1']}",
              f"- g2 unshifted change {result['unshifted_change_pp']:+.2f} pp (need >= -0.2): {result['checks']['g2']}",
              '- g3 every seed positive on the shrink mean ('+', '.join(f'{v:+.2f}' for v in result['shrink_gain_per_seed_pp'])+f"): {result['checks']['g3']}",
              '', 'Oracle rows take the window from the ground truth and are an upper bound for perfect localisation, not a method.',
              'Selection and gate use the same development data, so this is not a Test result.',
              'Part of the gain on shrunken images comes from the finer output grid of the zoomed pass (single-pass masks are',
              'produced at 224 and upsampled); the failure counts separate recovered lesions from sharper boundaries.', '']
    (REPORT/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return REPORT/'report.md'
