"""Protocol and paths of the box-crop control of SCZ. Evaluation only."""
import yaml

from research_formal.common import ROOT, WORKSPACE, now, write_json, sha256, object_hash
from research_detail_test.common import read, maybe, TARGETS

NAME = 'zoom_box_v1'
PROTOCOL = ROOT/'research_zoom_box/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_zoom_box'
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports'/f'{NAME}_results'
# The accepted window rule is max(sqrt(a/tau), margin*b, min_window). With this tau the first term is below 1e-6 for
# every possible area fraction a <= 1, i.e. below every min_window used here, so the rule is exactly max(margin*b, min_window).
NO_AREA_TERM = 1e12


def load_protocol():
    """Also verifies that the constants of the accepted method are unchanged."""
    from research_zoom.common import load_protocol as load_pilot
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    assert protocol['name'] == NAME
    assert (protocol['mc_samples'], protocol['batch_size'], protocol['threshold']) == (30, 32, .5)
    for relative, digest in protocol['lineage'].items():
        audit = read(WORKSPACE/relative)
        if sha256(WORKSPACE/relative) != digest or not audit['execution_valid'] or audit['gate_verdict'] != 'pass':
            raise ValueError('Acceptance record missing, changed or not passed: '+relative)
    pilot = load_pilot()
    if protocol['zoom'] != pilot['zoom'] or protocol['trigger'] != pilot['zoom']['zoom_in_if_window_at_most']:
        raise ValueError('The zoom constants differ from the accepted method')
    names = [v['name'] for v in protocol['variants']]
    assert len(set(names)) == len(names) and not set(names) & {'single', 't15'}
    assert sum(n.startswith('plain') for n in names) == 3
    for v in protocol['variants']:
        assert set(v) == {'name', 'margin', 'min_window'} and v['margin'] >= 1 and 1e-3 < v['min_window'] <= .2
    # in the form that research_zoom_probe.evaluate.evaluate_variants reads
    protocol['probe_variants'] = [dict(name=v['name'], tau=NO_AREA_TERM, margin=v['margin'], min_window=v['min_window'],
                                       trigger=protocol['trigger'], pixels='original', output='original')
                                  for v in protocol['variants']]
    protocol['seeds'] = [seed for seeds in protocol['sources'].values() for seed in seeds]
    assert protocol['seeds'] == [666, 667, 668, 669, 670]
    protocol['runs'] = [f"{protocol['arm']}_seed{seed}" for seed in protocol['seeds']]
    return protocol


def seed_of(run):
    return int(run.rpartition('_seed')[2])


def source_of(protocol, run):
    return next(name for name, seeds in protocol['sources'].items() if seed_of(run) in seeds)


def accepted_two_pass(protocol, run, target):
    return ROOT/'outputs_research'/protocol['accepted_two_pass'][source_of(protocol, run)]/'evaluations'/run/target/'cases.csv'


def job_keys(protocol):
    return ['check']+[f'box/{run}/{target}' for run in protocol['runs'] for target in TARGETS]+['report']


def code_manifest():
    files = sorted((ROOT/'research_zoom_box').rglob('*.py'))
    files += [ROOT/'research_zoom'/n for n in ('evaluate.py', 'geometry.py')]
    files += [ROOT/'research_zoom_probe'/n for n in ('evaluate.py', 'geometry.py')]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
