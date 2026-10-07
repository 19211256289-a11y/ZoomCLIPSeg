"""Paths, the locked protocol and resolution of the twelve accepted checkpoints."""
import json
from pathlib import Path

import yaml

from research_formal.common import ROOT, WORKSPACE, CKPT_NAME, now, write_json, sha256, object_hash

NAME = 'test_detail_mix_kvasir_v1'
MODES = ('baseline', 'global_mix', 'detail', 'detail_global_mix')
SEEDS = (666, 667, 668)
TARGETS = ('Kvasir', 'ColonDB', 'ClinicDB', 'CVC300', 'BKAI')
GENERIC = 'a polyp in a colonoscopy image'
TRAIN_SUITE = 'formal_detail_mix_kvasir_v1'
SMALL = 0.058989688326573426
RUNS = [f'{mode}_seed{seed}' for seed in SEEDS for mode in MODES]
COHORT_SIZES = dict(Val_check=100, Kvasir=100, ColonDB=360, ClinicDB=61, CVC300=60, BKAI=100)
PROTOCOL = ROOT/'research_detail_test/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_detail_test'/NAME
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports/test_detail_mix_20261004'
STEP0 = WORKSPACE/'reports/step0_val_case_analysis_20261004'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def maybe(path):
    return read(path) if Path(path).exists() else {}


def load_protocol():
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    assert protocol['name'] == NAME and protocol['training_suite'] == TRAIN_SUITE
    assert protocol['modes'] == list(MODES) and protocol['seeds'] == list(SEEDS)
    assert protocol['test_targets'] == list(TARGETS) and protocol['prompt'] == GENERIC
    assert (protocol['mc_samples'], protocol['batch_size'], protocol['threshold']) == (30, 32, .5)
    assert protocol['small_target_threshold'] == SMALL
    assert protocol['gate']['primary_contrast'] == ['detail', 'baseline']
    return protocol


def split_run(run):
    mode, marker, seed = run.rpartition('_seed')
    if not marker or mode not in MODES or int(seed) not in SEEDS:
        raise ValueError(run)
    return mode, int(seed)


def training_output(run):
    """Accepted final artifacts live in three verified backup roots (handoff section 12.2)."""
    mode, seed = split_run(run)
    backups = WORKSPACE/'server_backups'
    if seed == 666:
        root, host = backups/'server5090'/TRAIN_SUITE, 'local'
    elif mode == 'detail_global_mix':
        root, host = backups/'server5090/dg667_668', 'server4090'
    else:
        root, host = backups/'server4090'/TRAIN_SUITE, 'server4090'
    return root/'workspace/MedCLIPSeg/outputs_research'/TRAIN_SUITE/host


def checkpoint_path(run):
    return training_output(run)/run/'Kvasir/trained_models'/f'seed{split_run(run)[1]}'/CKPT_NAME


def val_cases_path(run):
    return training_output(run)/'evaluations'/run/'generic/cases.csv'


def accepted_runs(protocol):
    """Checkpoint hashes and Val summaries from the three hash-pinned acceptance audits."""
    runs = {}
    for relative, digest in protocol['accepted_audits'].items():
        path = WORKSPACE/relative
        if sha256(path) != digest:
            raise ValueError('Acceptance audit changed since the protocol was locked: '+relative)
        audit = read(path)
        if not audit['passed'] or set(runs) & set(audit['runs']):
            raise ValueError('Unexpected acceptance audit content: '+relative)
        runs.update(audit['runs'])
    if sorted(runs) != sorted(RUNS):
        raise ValueError('The audits do not cover exactly the twelve formal runs')
    for run, record in runs.items():
        if record['epoch'] != 100 or record['summary']['run_id'] != run:
            raise ValueError('Unexpected audit record for '+run)
    return runs


def code_manifest():
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'research_detail_test').rglob('*.py'))}
