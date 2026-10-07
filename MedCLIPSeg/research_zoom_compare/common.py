"""Protocol and paths of the comparison of SCZ with generic inference alternatives. Evaluation only."""
import yaml

from research_formal.common import ROOT, WORKSPACE, now, write_json, sha256, object_hash
from research_detail_test.common import read, maybe, TARGETS

NAME = 'zoom_compare_v1'
PROTOCOL = ROOT/'research_zoom_compare/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_zoom_compare'
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports'/f'{NAME}_results'
ALTERNATIVES = ('hflip', 'tile2', 'ms12', 'tile15', 'ms123')


def load_protocol():
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    assert protocol['name'] == NAME and tuple(protocol['alternatives']) == ALTERNATIVES
    assert (protocol['mc_samples'], protocol['batch_size'], protocol['threshold']) == (30, 32, .5)
    assert 9*protocol['images_per_batch'] <= protocol['batch_size']
    for relative, digest in protocol['lineage'].items():
        audit = read(WORKSPACE/relative)
        if sha256(WORKSPACE/relative) != digest or not audit['execution_valid'] or audit['gate_verdict'] != 'pass':
            raise ValueError('Acceptance record missing, changed or not passed: '+relative)
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
    return ['check']+[f'compare/{run}/{target}' for run in protocol['runs'] for target in TARGETS]+['report']


def code_manifest():
    files = sorted((ROOT/'research_zoom_compare').rglob('*.py'))
    files += [ROOT/'research_zoom'/n for n in ('evaluate.py', 'geometry.py')]+[ROOT/'research_zoom_probe/evaluate.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
