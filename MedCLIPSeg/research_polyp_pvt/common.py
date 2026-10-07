"""Protocol, paths and small helpers of the same-protocol Polyp-PVT reference (training and evaluation)."""
import json
import os
from pathlib import Path

import yaml

from research_formal.common import ROOT, WORKSPACE, now, write_json, sha256, object_hash
from research_detail_test.common import read, maybe

NAME = 'polyp_pvt_kvasir_v1'
PROTOCOL = ROOT/'research_polyp_pvt/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_polyp_pvt'
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports'/f'{NAME}_results'
WEIGHTS = ROOT/'checkpoints/pvt_v2_b2.pth'
TARGETS = ('Kvasir', 'ColonDB', 'ClinicDB', 'CVC300', 'BKAI')
OOD = TARGETS[1:]
HELD_OUT = 'ETIS'
COHORT_SIZES = dict(Kvasir=100, ColonDB=360, ClinicDB=61, CVC300=60, BKAI=100)
SIZES = (256, 352, 448)


class Paused(Exception):
    pass


class Control:
    """Pause request and progress file of this queue (same file names as the other queues)."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state = {}

    def pause_requested(self):
        return (self.directory/'PAUSE.request').exists()

    def check(self, **fields):
        if self.pause_requested():
            self.update('paused', **fields)
            raise Paused()

    def update(self, status, **fields):
        self.state.update(fields)
        self.state.update(status=status, pid=os.getpid(), updated_at=now())
        target = self.directory/'trainer_status.json'
        temporary = target.with_name(target.name+f'.tmp-{os.getpid()}')
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, target)


def load_protocol():
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    recipe = protocol['recipe']
    assert protocol['name'] == NAME and protocol['dataset'] == 'Kvasir' and protocol['seeds'] == [666, 667, 668]
    assert tuple(protocol['test_targets']) == TARGETS and tuple(protocol['ood_targets']) == OOD
    assert protocol['held_out_target'] == HELD_OUT and protocol['threshold'] == .5
    assert (recipe['input_size'], recipe['epochs'], recipe['batch_size']) == (352, 99, 16)
    assert (recipe['learning_rate'], recipe['weight_decay'], recipe['gradient_clip']) == (1e-4, 1e-4, .5)
    assert recipe['optimizer'] == 'AdamW' and recipe['precision'] == 'single'
    assert tuple(int(round(recipe['input_size']*rate/32)*32) for rate in recipe['scales']) == SIZES
    assert set(protocol['comparison']['ours']) == {'baseline', 'full'}
    return protocol


def accepted_cases(protocol, model, seed, target):
    """(file, value of the configuration column or None) of the accepted per-image results of one of our models."""
    suite, arm, config = protocol['comparison']['ours'][model]['held_out' if target == HELD_OUT else 'polyp']
    return ROOT/'outputs_research'/suite/'evaluations'/f'{arm}_seed{seed}'/target/'cases.csv', config


def training_dir(seed):
    return OUTPUT/'training'/f'seed{seed}'


def evaluation_dir(seed, target):
    return OUTPUT/'evaluations'/f'seed{seed}'/target


def job_keys(protocol):
    keys = ['check']
    for seed in protocol['seeds']:
        keys += [f'train/seed{seed}']+[f'eval/seed{seed}/{target}' for target in TARGETS+(HELD_OUT,)]
    return keys+['report']


def code_manifest():
    """Code that determines the numbers. The runner and the report (analysis of saved files) are not part of the seal."""
    package = ROOT/'research_polyp_pvt'
    files = sorted((package/'lib').glob('*.py'))+[package/n for n in ('common.py', 'data.py', 'train.py', 'evaluate.py')]
    files += [ROOT/'research_formal/metrics.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
