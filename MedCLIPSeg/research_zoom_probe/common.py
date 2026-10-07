"""Protocol and paths of the evaluation-only probe (review analyses). No Test-based selection happens here."""
from pathlib import Path

import yaml

from research_formal.common import ROOT, WORKSPACE, now, write_json, sha256, object_hash
from research_detail_test.common import read, maybe

NAME = 'zoom_probe_v1'
PROTOCOL = ROOT/'research_zoom_probe/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_zoom_probe'
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports'/f'{NAME}_results'
VARIANT_KEYS = ('tau', 'margin', 'min_window', 'trigger', 'pixels', 'output')


def load_protocol():
    """Also verifies that the reference configuration is exactly the accepted method."""
    from research_zoom.common import load_protocol as load_pilot
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    assert protocol['name'] == NAME
    assert (protocol['mc_samples'], protocol['batch_size'], protocol['threshold']) == (30, 32, .5)
    for relative, digest in protocol['lineage'].items():
        audit = read(WORKSPACE/relative)
        if sha256(WORKSPACE/relative) != digest or not audit['execution_valid'] or audit['gate_verdict'] != 'pass':
            raise ValueError('Acceptance record missing, changed or not passed: '+relative)
    pilot = load_pilot()
    accepted = next(c for c in pilot['configs'] if c['name'] == 't15')
    polyp, reference = protocol['polyp'], protocol['polyp']['reference']
    if protocol['zoom'] != pilot['zoom'] or accepted['oracle'] or accepted['zoom_out'] \
            or reference['tau'] != accepted['target_fraction'] or reference['margin'] != pilot['zoom']['margin'] \
            or reference['min_window'] != pilot['zoom']['min_window'] \
            or reference['trigger'] != pilot['zoom']['zoom_in_if_window_at_most'] \
            or (reference['pixels'], reference['output'], reference['name']) != ('original', 'original', 't15'):
        raise ValueError('The reference configuration differs from the accepted method')
    variants = [dict(reference, group='reference')]
    for item in polyp['variants']:
        unknown = set(item)-set(VARIANT_KEYS)-{'name', 'group'}
        if unknown or item['group'] not in polyp['groups']:
            raise ValueError(f'Bad variant: {item}')
        variants.append(dict(reference, **item))
    names = [v['name'] for v in variants]
    assert len(set(names)) == len(names) and 'single' not in names
    polyp['all_variants'] = variants
    seeds = [seed for group in polyp['sources'].values() for seed in group]
    assert seeds == polyp['groups']['resolution'] and set(polyp['groups']['sensitivity']) <= set(seeds)
    polyp['runs'] = [f"{polyp['arm']}_seed{seed}" for seed in seeds]
    ultra = protocol['ultrasound']
    ultra['runs'] = [f'{arm}_seed{seed}' for seed in ultra['seeds'] for arm in ultra['arms']]
    assert ultra['cohorts'][0] == 'Val'
    return protocol


def seed_of(run):
    return int(run.rpartition('_seed')[2])


def polyp_source(protocol, run):
    return next(name for name, seeds in protocol['polyp']['sources'].items() if seed_of(run) in seeds)


def variants_for(protocol, run):
    """The reference and every variant whose group includes the seed of this run."""
    groups = protocol['polyp']['groups']
    return [v for v in protocol['polyp']['all_variants'] if v['group'] == 'reference' or seed_of(run) in groups[v['group']]]


def job_keys(protocol):
    from research_detail_test.common import TARGETS
    keys = ['check']+[f'timing/{run}' for run in protocol['timing']['runs']]
    keys += [f'detect/{run}/{cohort}' for run in protocol['ultrasound']['runs'] for cohort in protocol['ultrasound']['cohorts']]
    keys += [f'zoom/{run}/{target}' for run in protocol['polyp']['runs'] for target in TARGETS]
    return keys+['report']


def code_manifest():
    files = sorted((ROOT/'research_zoom_probe').rglob('*.py'))+[ROOT/'research_zoom'/n for n in ('evaluate.py', 'geometry.py')]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
