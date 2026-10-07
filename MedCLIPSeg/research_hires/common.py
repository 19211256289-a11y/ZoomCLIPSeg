"""Suite, configuration and paths for the native-resolution pilot."""
import json
from pathlib import Path

import yaml

from research_formal.common import ROOT, WORKSPACE, CKPT_NAME, now, write_json, sha256, object_hash, plain_config

NAME = 'pilot_hires_kvasir_v1'
SUITE_FILE = ROOT/'research_hires/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_hires'/NAME
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports'/f'{NAME}_results'
HIRES = 448


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def maybe(path):
    return read(path) if Path(path).exists() else {}


def load_suite():
    suite = yaml.safe_load(SUITE_FILE.read_text(encoding='utf-8'))
    assert suite['name'] == NAME and suite['dataset'] == 'Kvasir' and suite['epochs'] == 30
    assert list(suite['arms']) == ['baseline', 'detail224', 'sup448', 'detail448']
    assert (suite['mc_samples'], suite['batch_size'], suite['threshold']) == (30, 32, .5)
    suite['modes'] = list(suite['arms'])
    suite['runs'] = [f'{arm}_seed{seed}' for seed in suite['seeds'] for arm in suite['modes']]
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


def make_config(suite, run):
    """The accepted Kvasir configuration; only epochs, seed, bookkeeping and the resolution mode change."""
    from research_detail_formal import common as formal
    arm, seed = split_run(suite, run)
    spec = suite['arms'][arm]
    cfg = formal.make_config(formal.load_suite(), f"{spec['query_mode']}_seed666")
    cfg.TRAIN.NUM_EPOCHS = suite['epochs']
    cfg.RESEARCH.SUITE = suite['name']
    cfg.RESEARCH.RUN_ID = run
    cfg.RESEARCH.INITIALIZATION_SEED = seed
    cfg.RESEARCH.DETAIL_INITIALIZATION_SEED = seed+448
    cfg.RESEARCH.GENERIC_PROMPT = suite['prompt']
    cfg.RESEARCH.HIRES_MODE = spec['hires']
    cfg.RESEARCH.PROTOCOL_SHA256 = sha256(SUITE_FILE)
    cfg.seed = seed
    cfg.output_dir = str(OUTPUT/run)
    cfg.control_dir = str(CONTROL)
    cfg.resume = False
    return cfg


def science_config(cfg):
    return {key: plain_config(cfg[key]) for key in ('DATASET', 'TRAIN', 'TEST', 'MODEL', 'RESEARCH')}


def code_manifest():
    """Files that define the experiment itself. Queue, evaluation wrapper and report code are tooling."""
    paths = [ROOT/'research_hires'/name for name in ('common.py', 'model.py', 'data.py')]+[ROOT/'train_hires.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}
