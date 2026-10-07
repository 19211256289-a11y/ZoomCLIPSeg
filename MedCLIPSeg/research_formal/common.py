"""Paths, configuration, atomic state, and auditable research signatures."""
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CKPT_NAME = 'MedCLIPSeg_unimedclip_ViT-B-16_latest.pth'
TARGETS = {'Kvasir': ['ColonDB', 'ClinicDB', 'CVC300', 'BKAI'],
           'BUSI': ['BUSBRA', 'BUSUC', 'BUID', 'UDIAT'], 'Covid19': []}
VARIANTS = ('paired_repeat', 'paired_image', 'paired_aug')


def split_run(run):
    variant, marker, seed = run.rpartition('_seed')
    if not marker or variant not in VARIANTS or seed not in ('666', '667', '668'):
        raise ValueError(f'Unknown Formal run: {run}')
    return variant, int(seed)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_suite(name):
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name):
        raise ValueError('Suite must be a lowercase preset name')
    path = ROOT / 'research_formal/suites' / f'{name}.yaml'
    suite = yaml.safe_load(path.read_text(encoding='utf-8'))
    if suite['dataset'] not in TARGETS or not suite['variants']:
        raise ValueError('Unsupported source dataset or empty suite')
    if len(set(suite['variants'])) != len(suite['variants']) or any(v not in VARIANTS for v in suite['variants']):
        raise ValueError('Duplicate/unknown variant in suite')
    if suite['percentage'] != 100 or suite['epochs'] != 100:
        raise ValueError('Formal suites require 100% data and exactly 100 epochs')
    if suite['dataset'] != 'Kvasir' or suite['variants'] != list(VARIANTS) or suite['seeds'] != [666, 667, 668]:
        raise ValueError('Formal fixes three matched-view arms and three training RNG seeds')
    if suite['learning_rate'] != 3e-4 or suite['min_learning_rate'] != 1e-4:
        raise ValueError('Formal fixes the original from-base learning rates')
    if (suite['initialization'] != 'unimedclip_base_fresh_adapters_and_head_per_seed'
            or suite['checkpoint_selection'] != 'final_epoch' or suite['evaluation_mc'] != 30
            or suite['test_targets'] != ['Kvasir'] + TARGETS['Kvasir']
            or suite['automatic_test_after_all_training'] is not True):
        raise ValueError('Formal initialization/final evaluation protocol changed')
    # Rotate arm order across seeds to reduce confounding from run order.
    suite['runs'] = [f'{VARIANTS[(arm+index)%3]}_seed{seed}'
                     for index, seed in enumerate(suite['seeds']) for arm in range(3)]
    suite['name'] = name
    return suite


def control_dir(suite):
    return WORKSPACE / 'control/research_formal' / suite['name']


def suite_output(suite):
    return ROOT / 'outputs_research' / suite['name']


def checkpoint_path(suite, run):
    _, seed = split_run(run)
    return suite_output(suite) / run / suite['dataset'] / 'trained_models' / f'seed{seed}' / CKPT_NAME


def make_config(suite, run, checkpointing=False):
    from utils.main_utils import load_cfg_from_cfg_file, CfgNode
    variant, seed = split_run(run)
    if run not in suite['runs']:
        raise ValueError(run)
    cfg = load_cfg_from_cfg_file(str(ROOT / 'configs' / f"{suite['dataset']}.yaml"))
    cfg.MODEL.VARIANT = variant
    cfg.MODEL.GRAD_CHECKPOINTING = checkpointing
    cfg.TRAIN.CONSISTENCY_WEIGHT = 0.0
    cfg.DATASET.DETAIL_SIZE = 448
    for key in ('TRAIN_PATH', 'VAL_PATH', 'TEST_PATH', 'TEXT_PROMPT_PATH'):
        cfg.DATASET[key] = str((ROOT / cfg.DATASET[key]).resolve()) + os.sep
    cfg.TRAIN.NUM_EPOCHS = suite['epochs']
    cfg.TRAIN.LEARNING_RATE = suite['learning_rate']
    cfg.TRAIN.NEW_LEARNING_RATE = suite['learning_rate']
    cfg.TRAIN.MIN_LEARNING_RATE = suite['min_learning_rate']
    cfg.TRAIN.DICE_MODE = 'image'
    cfg.TRAIN.BETA_WEIGHT = 1e-3
    cfg.TRAIN.AUX_WEIGHT = 0.
    cfg.TRAIN.HARD_BG_FRACTION = .1
    cfg.RESEARCH = CfgNode({'BASE_WEIGHTS': str(ROOT/'checkpoints/unimed_clip_vit_b16.pt'),
                           'INITIALIZATION': suite['initialization'], 'SUITE': suite['name'], 'RUN_ID': run,
                           'INITIALIZATION_SEED': seed,
                           'PROTOCOL_SHA256': sha256(ROOT / 'research_formal/suites' / f"{suite['name']}.yaml")})
    cfg.seed, cfg.data_percentage, cfg.num_workers = seed, suite['percentage'], 0
    cfg.output_dir = str(suite_output(suite) / run)
    cfg.control_dir = str(control_dir(suite))
    cfg.resume = False
    cfg.init_from = None
    return cfg


def plain_config(cfg):
    return json.loads(json.dumps(cfg))


def code_manifest():
    fixed = ['train_formal.py', 'datasets/dataloader.py', 'local_runtime.py', 'utils/main_utils.py',
             'utils/SurfaceDice.py', 'trainers/medclipseg_unimedclip.py', 'trainers/layers.py',
             'trainers/scale_block.py']
    paths = [ROOT / p for p in fixed] + sorted((ROOT / 'research_formal').glob('*.py'))
    paths += sorted((ROOT / 'open_clip_lib').rglob('*.py'))
    paths += sorted((ROOT / 'open_clip_lib/model_configs').glob('*.json'))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def data_manifest(cfg):
    from utils.main_utils import read_text
    entries = {}
    for split in ('Train', 'Val'):
        suffix = '' if cfg.data_percentage == 100 else f'_{cfg.data_percentage}'
        text_path = Path(cfg.DATASET.TEXT_PROMPT_PATH) / f'{split}_text{suffix}.xlsx'
        entries[str(text_path)] = sha256(text_path)
        folder = Path(getattr(cfg.DATASET, split.upper() + '_PATH'))
        for record in read_text(text_path):
            for kind, key in [('img', 'Image'), ('label', 'Ground Truth')]:
                path = folder / kind / record[key]
                if str(path) not in entries:
                    entries[str(path)] = sha256(path)
    return entries


def make_signature(cfg):
    import torch
    import monai
    if getattr(cfg, 'init_from', None):
        raise ValueError('Formal training forbids warm-start checkpoints; use the locked base weights')
    init_path = cfg.RESEARCH.BASE_WEIGHTS
    values = {k: plain_config(cfg[k]) for k in ('MODEL', 'TRAIN', 'DATASET', 'TEST', 'RESEARCH')}
    values.update(seed=cfg.seed, percentage=cfg.data_percentage, workers=cfg.num_workers)
    signature = dict(config=values, init_sha256=sha256(init_path),
                     code=code_manifest(), data=data_manifest(cfg), assets=asset_manifest(),
                     environment={'python': platform.python_version(), 'torch': torch.__version__,
                                  'cuda': torch.version.cuda, 'monai': monai.__version__,
                                  'device': torch.cuda.get_device_name() if cfg.MODEL.DEVICE == 'cuda' else 'cpu'})
    signature['digest'] = object_hash(signature)
    return signature


def asset_manifest():
    # Include the local text encoder configuration/tokenizer as well as its weights.
    folder = WORKSPACE/'.cache/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract'
    files = sorted(p for p in folder.rglob('*') if p.is_file() and ('snapshots' in p.parts or 'refs' in p.parts))
    if not files:
        raise FileNotFoundError('Local BiomedBERT assets are missing')
    return {str(p): sha256(p) for p in files}


def test_cohort_manifest(suite):
    """Seal input files and check disjointness without computing any test performance."""
    from utils.main_utils import read_text
    source = make_config(suite, suite['runs'][0], True)
    training_files = data_manifest(source)
    source_hashes = {v for p, v in training_files.items() if Path(p).parent.name == 'img'}
    cohorts = {}
    for target in suite['test_targets']:
        prompt = ROOT/'data'/target/'Prompts_Folder/Test_text_original.xlsx'
        rows = read_text(prompt)
        if not rows or len({r['Image'] for r in rows}) != len(rows) or len({r['Ground Truth'] for r in rows}) != len(rows):
            raise ValueError(f'Empty or duplicate test manifest: {target}')
        files, image_hashes, groups = {}, [], {}
        for row in rows:
            for kind, key in [('img', 'Image'), ('label', 'Ground Truth')]:
                path = ROOT/'data'/target/'Test_Folder'/kind/row[key]
                if path.stat().st_size == 0:
                    raise ValueError(f'Empty file: {path}')
                files[str(path)] = sha256(path)
                if kind == 'img':
                    image_hashes.append(files[str(path)])
                    groups.setdefault(files[str(path)], []).append(row['Image'])
        if source_hashes.intersection(image_hashes):
            raise ValueError(f'Test images overlap source Train/Val: {target}')
        cohorts[target] = dict(n=len(rows), n_unique_images=len(groups), prompt_sha256=sha256(prompt), files=files,
                               image_hash_groups=groups, duplicate_image_groups=[v for v in groups.values() if len(v)>1])
    for target, cohort in cohorts.items():
        cohort['shared_unique_images_with_other_tests'] = {
            other: len(set(cohort['image_hash_groups']) & set(other_cohort['image_hash_groups']))
            for other, other_cohort in cohorts.items() if other != target
            and set(cohort['image_hash_groups']) & set(other_cohort['image_hash_groups'])}
    return cohorts


def assert_signature(saved, current):
    if not saved or saved.get('digest') != current['digest']:
        raise ValueError('Research signature changed (code/config/data/environment/init weights). Start a new experiment; do not resume this checkpoint.')


def optimizer_groups(model, cfg):
    from .model import is_new_parameter
    existing, added = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (added if is_new_parameter(name) else existing).append(parameter)
    result = [{'params': existing, 'lr': cfg.TRAIN.LEARNING_RATE}]
    if added:
        result.append({'params': added, 'lr': cfg.TRAIN.NEW_LEARNING_RATE})
    return result
