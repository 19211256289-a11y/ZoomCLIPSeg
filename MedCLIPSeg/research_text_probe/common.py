"""Protocol and paths of the test-time text analysis, the one-pass run and the timing of the fixed-sentence models.
Evaluation only."""
import yaml

from research_formal.common import ROOT, WORKSPACE, now, write_json, sha256, object_hash
from research_detail_test.common import read, maybe, TARGETS

NAME = 'text_probe_v1'
PROTOCOL = ROOT/'research_text_probe/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_text_probe'
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports'/f'{NAME}_results'
TEXTS = ('fixed', 'empty', 'wrong_class', 'paraphrase_a', 'paraphrase_b')      # 30 stochastic passes each
CONFIGS = TEXTS+('single01',)
TIMED = (1, 5, 15, 30)


def load_protocol():
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    assert protocol['name'] == NAME and protocol['arm'] == 'detail'
    assert (protocol['mc_samples'], protocol['batch_size'], protocol['threshold']) == (30, 32, .5)
    assert set(protocol['prompts']) == set(TEXTS[1:]) | {'empty_fallback'} and protocol['prompts']['empty'] == ''
    assert tuple(protocol['timing']['passes']) == TIMED
    for relative, digest in protocol['lineage'].items():
        audit = read(WORKSPACE/relative)
        if sha256(WORKSPACE/relative) != digest or not audit['execution_valid'] or audit['gate_verdict'] != 'pass':
            raise ValueError('Acceptance record missing, changed or not passed: '+relative)
    protocol['seeds'] = [seed for seeds in protocol['sources'].values() for seed in seeds]
    assert protocol['seeds'] == [666, 667, 668]
    protocol['runs'] = [f"{protocol['arm']}_seed{seed}" for seed in protocol['seeds']]
    assert protocol['timing']['model'] == protocol['runs'][0] and protocol['timing']['dataset'] == TARGETS[0]
    return protocol


def seed_of(run):
    return int(run.rpartition('_seed')[2])


def source_of(protocol, run):
    return next(name for name, seeds in protocol['sources'].items() if seed_of(run) in seeds)


def job_keys(protocol):
    return ['check']+[f'probe/{run}/{target}' for run in protocol['runs'] for target in TARGETS]+['timing', 'report']


def code_manifest():
    files = sorted((ROOT/'research_text_probe').rglob('*.py'))
    files += [ROOT/'research_zoom/evaluate.py', ROOT/'research_zoom_probe/evaluate.py', ROOT/'research_formal/metrics.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
