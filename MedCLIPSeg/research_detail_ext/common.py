"""Suites, configuration, paths and sealed evaluation cohorts for the extension campaign."""
import json
import os
from pathlib import Path

import yaml

from research_formal.common import ROOT, WORKSPACE, CKPT_NAME, now, write_json, sha256, object_hash, plain_config

SUITES = ('formal_detail_mix_busi_v1', 'formal_detail_mix_kvasir_v2')
ALL_MODES = ('baseline', 'global_mix', 'detail', 'detail_global_mix')
CONTROL = WORKSPACE/'control/research_detail_ext'
PARENT_AUDIT = WORKSPACE/'reports/acceptance_test_detail_mix_20261004/audit.json'
KVASIR_SMALL = 0.058989688326573426
FOLDERS = (('TRAIN_PATH', 'Train_Folder'), ('VAL_PATH', 'Val_Folder'), ('TEST_PATH', 'Test_Folder'),
           ('TEXT_PROMPT_PATH', 'Prompts_Folder'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def maybe(path):
    return read(path) if Path(path).exists() else {}


def suite_file(name):
    return ROOT/'research_detail_ext/suites'/f'{name}.yaml'


def load_suite(name):
    if name not in SUITES:
        raise ValueError(name)
    suite = yaml.safe_load(suite_file(name).read_text(encoding='utf-8'))
    assert suite['name'] == name and suite['epochs'] == 100
    assert all(mode in ALL_MODES for mode in suite['modes']) and suite['modes'][0] == 'baseline'
    assert (suite['mc_samples'], suite['batch_size'], suite['threshold']) == (30, 32, .5)
    assert suite['test_targets'][0] == suite['dataset'] and suite['ood_targets'] == suite['test_targets'][1:]
    assert isinstance(suite['prompt'], str) and suite['prompt'].strip()
    suite['runs'] = [f'{mode}_seed{seed}' for seed in suite['seeds'] for mode in suite['modes']]
    return suite


def split_run(suite, run):
    mode, marker, seed = run.rpartition('_seed')
    if not marker or mode not in suite['modes'] or int(seed) not in suite['seeds']:
        raise ValueError(run)
    return mode, int(seed)


def output(suite):
    return ROOT/'outputs_research'/suite['name']


def report_dir(suite):
    return WORKSPACE/'reports'/f"{suite['name']}_results"


def checkpoint_path(suite, run):
    return output(suite)/run/suite['dataset']/'trained_models'/f'seed{split_run(suite, run)[1]}'/CKPT_NAME


def config_path(suite, run):
    return output(suite)/'configs'/f'{run}.yaml'


def make_config(suite, run):
    """The accepted Kvasir configuration with only dataset, seed, sentence and bookkeeping changed."""
    from research_detail_formal import common as formal
    mode, seed = split_run(suite, run)
    cfg = formal.make_config(formal.load_suite(), f'{mode}_seed666')
    if suite['dataset'] != 'Kvasir':
        cfg.DATASET.NAME = suite['dataset']
        for key, folder in FOLDERS:
            cfg.DATASET[key] = str((ROOT/'data'/suite['dataset']/folder).resolve())+os.sep
    cfg.RESEARCH.SUITE = suite['name']
    cfg.RESEARCH.RUN_ID = run
    cfg.RESEARCH.INITIALIZATION_SEED = seed
    cfg.RESEARCH.DETAIL_INITIALIZATION_SEED = seed+448
    cfg.RESEARCH.GENERIC_PROMPT = suite['prompt']
    cfg.RESEARCH.PROTOCOL_SHA256 = sha256(suite_file(suite['name']))
    cfg.RESEARCH.PARENT_ACCEPTANCE_SHA256 = sha256(PARENT_AUDIT)
    cfg.seed = seed
    cfg.output_dir = str(output(suite)/run)
    cfg.control_dir = str(CONTROL)
    cfg.resume = False
    return cfg


def science_config(cfg):
    return {key: plain_config(cfg[key]) for key in ('DATASET', 'TRAIN', 'TEST', 'MODEL', 'RESEARCH')}


def tooling_manifest():
    paths = sorted((ROOT/'research_detail_ext').rglob('*.py'))+sorted((ROOT/'research_detail_test').glob('stats.py'))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def source_image_hashes(cfg):
    from utils.main_utils import read_text
    hashes = set()
    for split, key in (('Train', 'TRAIN_PATH'), ('Val', 'VAL_PATH')):
        for row in read_text(Path(cfg.DATASET.TEXT_PROMPT_PATH)/f'{split}_text.xlsx'):
            hashes.add(sha256(Path(cfg.DATASET[key])/'img'/row['Image']))
    return hashes


def cohort_manifest(suite, cfg):
    """Seal the complete test lists and files; computes no performance number."""
    from utils.main_utils import read_text
    source = source_image_hashes(cfg)
    cohorts = {}
    for target in suite['test_targets']:
        listing = ROOT/'data'/target/'Prompts_Folder'/suite['test_list']
        rows = read_text(listing)
        if not rows or len({r['Image'] for r in rows}) != len(rows) or len({r['Ground Truth'] for r in rows}) != len(rows):
            raise ValueError(f'Empty or duplicate test list: {target}')
        files, groups = {}, {}
        for row in rows:
            for kind, key in (('img', 'Image'), ('label', 'Ground Truth')):
                path = ROOT/'data'/target/'Test_Folder'/kind/row[key]
                if path.stat().st_size == 0:
                    raise ValueError(f'Empty file: {path}')
                files[str(path)] = sha256(path)
                if kind == 'img':
                    groups.setdefault(files[str(path)], []).append(row['Image'])
        cohorts[target] = dict(n=len(rows), n_unique_images=len(groups), list_sha256=sha256(listing), files=files,
                               image_hash_groups=groups,
                               overlap_with_source_train_val=sorted(n for h, names in groups.items() if h in source for n in names))
    for target, cohort in cohorts.items():
        cohort['shared_unique_images_with_other_tests'] = {
            other: len(set(cohort['image_hash_groups']) & set(c['image_hash_groups']))
            for other, c in cohorts.items() if other != target and set(cohort['image_hash_groups']) & set(c['image_hash_groups'])}
    return cohorts
