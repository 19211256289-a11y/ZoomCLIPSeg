"""Suite, configuration and paths of the reduced-training-set experiments."""
import json
from pathlib import Path

import yaml

from research_formal.common import ROOT, WORKSPACE, CKPT_NAME, now, write_json, sha256, object_hash, plain_config
from research_detail_test.common import SMALL, TARGETS

NAME = 'lowdata_kvasir_v1'
SUITE_FILE = ROOT/'research_lowdata/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_lowdata'
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports'/f'{NAME}_results'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def maybe(path):
    return read(path) if Path(path).exists() else {}


def load_suite():
    suite = yaml.safe_load(SUITE_FILE.read_text(encoding='utf-8'))
    assert suite['name'] == NAME and suite['dataset'] == 'Kvasir' and suite['epochs'] == 100 and suite['seeds'] == [666, 667, 668]
    assert suite['fractions'] == [10, 25] and list(suite['arms']) == ['baseline', 'detail']
    assert (suite['mc_samples'], suite['batch_size'], suite['threshold']) == (30, 32, .5)
    assert suite['small_target_threshold'] == SMALL and tuple(suite['test_targets']) == TARGETS
    assert suite['ood_targets'] == suite['test_targets'][1:]
    for key in ('training_images', 'validation_images'):
        assert list(suite[key]) == [f'p{fraction}' for fraction in suite['fractions']], key
    audit = WORKSPACE/suite['two_pass']['acceptance_audit']
    if sha256(audit) != suite['two_pass']['acceptance_sha256'] or read(audit)['gate_verdict'] != 'pass':
        raise ValueError('The two-pass acceptance record is missing, changed or not passed')
    suite['runs'] = [f'{arm}_p{fraction}_seed{seed}' for fraction in suite['fractions'] for arm in suite['arms']
                     for seed in suite['seeds']]          # the smaller fraction first
    return suite


def split_run(suite, run):
    head, marker, seed = run.rpartition('_seed')
    arm, marker2, fraction = head.rpartition('_p')
    if not marker or not marker2 or arm not in suite['arms'] or int(seed) not in suite['seeds'] or int(fraction) not in suite['fractions']:
        raise ValueError(run)
    return arm, int(fraction), int(seed)


def checkpoint_path(suite, run):
    arm, fraction, seed = split_run(suite, run)
    return OUTPUT/run/f"{suite['dataset']}_{fraction}"/'trained_models'/f'seed{seed}'/CKPT_NAME


def config_path(run):
    return OUTPUT/'configs'/f'{run}.yaml'


def training_list(suite, fraction):
    return ROOT/'data'/suite['dataset']/'Prompts_Folder'/f'Train_text_{fraction}.xlsx'


def groups(suite):
    """(fraction, arm, runs) in the order of execution: the smaller fraction first, arm by arm."""
    return [(fraction, arm, [f'{arm}_p{fraction}_seed{seed}' for seed in suite['seeds']])
            for fraction in suite['fractions'] for arm in suite['arms']]


def job_keys(suite):
    keys = ['check']
    for _, _, runs in groups(suite):
        keys += [f'train/{run}' for run in runs]+[f'eval/{run}/{target}' for run in runs for target in suite['test_targets']]
    return keys+['report']


def code_manifest():
    """Files that define the experiments besides the accepted trainer. Queue, evaluation wrapper and report are tooling."""
    paths = [ROOT/'research_lowdata/common.py', ROOT/'train_controls.py', ROOT/'research_controls/recipes.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def make_config(suite, run):
    """The accepted Kvasir configuration; only seed, bookkeeping and the training fraction change."""
    from research_detail_formal import common as formal
    arm, fraction, seed = split_run(suite, run)
    cfg = formal.make_config(formal.load_suite(), f"{suite['arms'][arm]['query_mode']}_seed666")
    assert cfg.TRAIN.NUM_EPOCHS == suite['epochs'] and cfg.TRAIN.DICE_MODE == 'image'
    cfg.RESEARCH.SUITE = suite['name']
    cfg.RESEARCH.RUN_ID = run
    cfg.RESEARCH.INITIALIZATION_SEED = seed
    cfg.RESEARCH.DETAIL_INITIALIZATION_SEED = seed+448
    cfg.RESEARCH.GENERIC_PROMPT = suite['prompt']
    cfg.RESEARCH.CONTROL_ARM = arm
    cfg.RESEARCH.CONTROL_RECIPE = 'paired'
    cfg.RESEARCH.CONTROL_DATA = 'standard'
    cfg.RESEARCH.TRAINING_FRACTION = fraction
    cfg.RESEARCH.CONTROL_CODE_SHA256 = object_hash(code_manifest())
    cfg.RESEARCH.PROTOCOL_SHA256 = sha256(SUITE_FILE)
    cfg.seed = seed
    cfg.output_dir = str(OUTPUT/run)
    cfg.control_dir = str(CONTROL)
    cfg.resume = False
    return cfg


def science_config(cfg):
    return {key: plain_config(cfg[key]) for key in ('DATASET', 'TRAIN', 'TEST', 'MODEL', 'RESEARCH')}
