"""Suite, configuration and paths for the control experiments."""
import json
from pathlib import Path

import yaml

from research_formal.common import ROOT, WORKSPACE, CKPT_NAME, now, write_json, sha256, object_hash, plain_config

NAME = 'formal_controls_kvasir_v1'
SUITE_FILE = ROOT/'research_controls/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_controls'/NAME
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports'/f'{NAME}_results'
RECIPES, DATA, DICE = ('paired', 'single', 'single_flip'), ('standard', 'scale_aug', 'upsampled_detail'), ('image', 'legacy')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def maybe(path):
    return read(path) if Path(path).exists() else {}


def load_suite():
    suite = yaml.safe_load(SUITE_FILE.read_text(encoding='utf-8'))
    assert suite['name'] == NAME and suite['dataset'] == 'Kvasir' and suite['epochs'] == 100 and suite['seeds'] == [666, 667, 668]
    assert (suite['mc_samples'], suite['batch_size'], suite['threshold']) == (30, 32, .5)
    for arm, spec in suite['arms'].items():
        assert spec['query_mode'] in ('baseline', 'detail') and spec['recipe'] in RECIPES, arm
        assert spec['data'] in DATA and spec['dice'] in DICE and isinstance(spec['two_pass'], bool), arm
        assert spec['data'] == 'standard' or spec['query_mode'] == 'detail', arm
    for name, (a, ca, b, cb) in suite['contrasts'].items():
        for arm, config in ((a, ca), (b, cb)):
            assert arm in suite['arms'] and config in ('single', 't15'), name
            assert config == 'single' or suite['arms'][arm]['two_pass'], name
    audit = WORKSPACE/suite['two_pass']['acceptance_audit']
    if sha256(audit) != suite['two_pass']['acceptance_sha256'] or read(audit)['gate_verdict'] != 'pass':
        raise ValueError('The two-pass acceptance record is missing, changed or not passed')
    suite['runs'] = [f'{arm}_seed{seed}' for arm in suite['arms'] for seed in suite['seeds']]   # arm by arm
    return suite


def split_run(suite, run):
    arm, marker, seed = run.rpartition('_seed')
    if not marker or arm not in suite['arms'] or int(seed) not in suite['seeds']:
        raise ValueError(run)
    return arm, int(seed)


def checkpoint_path(suite, run):
    return OUTPUT/run/suite['dataset']/'trained_models'/f'seed{split_run(suite, run)[1]}'/CKPT_NAME


def config_path(run):
    return OUTPUT/'configs'/f'{run}.yaml'


def code_manifest():
    """Files that define the experiments. Queue, evaluation wrapper and report code are tooling."""
    paths = [ROOT/'research_controls'/name for name in ('common.py', 'recipes.py')]+[ROOT/'train_controls.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def make_config(suite, run):
    """The accepted Kvasir configuration; only seed, bookkeeping and the arm's recipe change."""
    from research_detail_formal import common as formal
    arm, seed = split_run(suite, run)
    spec = suite['arms'][arm]
    cfg = formal.make_config(formal.load_suite(), f"{spec['query_mode']}_seed666")
    assert cfg.TRAIN.NUM_EPOCHS == suite['epochs'] and cfg.TRAIN.DICE_MODE == 'image'
    cfg.TRAIN.DICE_MODE = spec['dice']
    cfg.RESEARCH.SUITE = suite['name']
    cfg.RESEARCH.RUN_ID = run
    cfg.RESEARCH.INITIALIZATION_SEED = seed
    cfg.RESEARCH.DETAIL_INITIALIZATION_SEED = seed+448
    cfg.RESEARCH.GENERIC_PROMPT = suite['prompt']
    cfg.RESEARCH.CONTROL_ARM = arm
    cfg.RESEARCH.CONTROL_RECIPE = spec['recipe']
    cfg.RESEARCH.CONTROL_DATA = spec['data']
    cfg.RESEARCH.CONTROL_CODE_SHA256 = object_hash(code_manifest())
    cfg.RESEARCH.PROTOCOL_SHA256 = sha256(SUITE_FILE)
    cfg.seed = seed
    cfg.output_dir = str(OUTPUT/run)
    cfg.control_dir = str(CONTROL)
    cfg.resume = False
    return cfg


def science_config(cfg):
    return {key: plain_config(cfg[key]) for key in ('DATASET', 'TRAIN', 'TEST', 'MODEL', 'RESEARCH')}
