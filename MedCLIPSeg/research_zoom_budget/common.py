"""Protocol and paths of the accuracy-against-budget analysis of SCZ and reflection averaging. Evaluation only."""
import yaml

from research_formal.common import ROOT, WORKSPACE, now, write_json, sha256, object_hash
from research_detail_test.common import read, maybe, TARGETS

NAME = 'zoom_budget_v1'
PROTOCOL = ROOT/'research_zoom_budget/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_zoom_budget'
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports'/f'{NAME}_results'
PASSES = (5, 10, 15, 30)
SINGLE = tuple(f'single{k:02d}' for k in PASSES)
HFLIP = ('hflip10', 'hflip15', 'hflip30')
SCZ = ('scz10', 'scz15')
SCZFLIP = ('sczflip10', 'sczflip15', 'sczflip30')
CONFIGS = SINGLE+HFLIP+SCZ+SCZFLIP          # computed here; 't15' (SCZ with 30 passes) is read from the accepted results


def passes_of(config):
    return 30 if config == 't15' else int(config[-2:])


def load_protocol():
    """Also verifies that the constants of the accepted method are unchanged."""
    from research_zoom.common import load_protocol as load_pilot
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    assert protocol['name'] == NAME and tuple(protocol['passes']) == PASSES
    assert (protocol['mc_samples'], protocol['batch_size'], protocol['threshold']) == (30, 32, .5)
    for relative, digest in protocol['lineage'].items():
        audit = read(WORKSPACE/relative)
        if sha256(WORKSPACE/relative) != digest or not audit['execution_valid'] or audit['gate_verdict'] != 'pass':
            raise ValueError('Acceptance record missing, changed or not passed: '+relative)
    pilot = load_pilot()
    accepted = next(c for c in pilot['configs'] if c['name'] == 't15')
    if protocol['zoom'] != pilot['zoom'] or protocol['target_fraction'] != accepted['target_fraction'] \
            or accepted['oracle'] or accepted['zoom_out']:
        raise ValueError('The zoom differs from the accepted method')
    for name, (a, b) in protocol['contrasts'].items():
        assert a in CONFIGS+('t15',) and b in CONFIGS+('t15',), name
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
    return ['check']+[f'budget/{run}/{target}' for run in protocol['runs'] for target in TARGETS]+['report']


def code_manifest():
    files = sorted((ROOT/'research_zoom_budget').rglob('*.py'))
    files += [ROOT/'research_zoom'/n for n in ('evaluate.py', 'geometry.py')]+[ROOT/'research_zoom_probe/evaluate.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
