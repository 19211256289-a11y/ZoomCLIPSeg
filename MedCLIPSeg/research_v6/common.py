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
VARIANTS = ('paired_aug', 'paired_consistency')


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
    path = ROOT / 'research_v6/suites' / f'{name}.yaml'
    suite = yaml.safe_load(path.read_text(encoding='utf-8'))
    if suite['dataset'] not in TARGETS or not suite['variants']:
        raise ValueError('Unsupported source dataset or empty suite')
    if len(set(suite['variants'])) != len(suite['variants']) or any(v not in VARIANTS for v in suite['variants']):
        raise ValueError('Duplicate/unknown variant in suite')
    if suite['percentage'] != 100 or suite['epochs'] != 20:
        raise ValueError('Warm-start pilot suites require 100% data and exactly 20 epochs')
    if suite.get('epochs_by_variant') != {'paired_aug': 20, 'paired_consistency': 20}:
        raise ValueError('V6 fixes paired augmentation and consistency with identical view budgets')
    suite['name'] = name
    return suite


def control_dir(suite):
    return WORKSPACE / 'control/research_v6' / suite['name']


def suite_output(suite):
    return ROOT / 'outputs_research' / suite['name']


def checkpoint_path(suite, variant):
    return suite_output(suite) / variant / suite['dataset'] / 'trained_models' / f"seed{suite['seed']}" / CKPT_NAME


def make_config(suite, variant, checkpointing=False):
    from utils.main_utils import load_cfg_from_cfg_file, CfgNode
    if variant not in VARIANTS:
        raise ValueError(variant)
    cfg = load_cfg_from_cfg_file(str(ROOT / 'configs' / f"{suite['dataset']}.yaml"))
    cfg.MODEL.VARIANT = variant
    cfg.MODEL.GRAD_CHECKPOINTING = checkpointing
    cfg.TRAIN.CONSISTENCY_WEIGHT = suite['consistency_weight']
    cfg.DATASET.DETAIL_SIZE = 448
    for key in ('TRAIN_PATH', 'VAL_PATH', 'TEST_PATH', 'TEXT_PROMPT_PATH'):
        cfg.DATASET[key] = str((ROOT / cfg.DATASET[key]).resolve()) + os.sep
    cfg.TRAIN.NUM_EPOCHS = suite['epochs_by_variant'][variant]
    cfg.TRAIN.LEARNING_RATE = 1e-4
    cfg.TRAIN.NEW_LEARNING_RATE = 1e-4
    cfg.TRAIN.MIN_LEARNING_RATE = 1e-5
    cfg.TRAIN.DICE_MODE = 'image'
    cfg.TRAIN.BETA_WEIGHT = 1e-3
    cfg.TRAIN.AUX_WEIGHT = 0.
    cfg.TRAIN.HARD_BG_FRACTION = .1
    init = ROOT / 'outputs_local_training' / suite['dataset'] / 'trained_models/seed666' / CKPT_NAME
    cfg.RESEARCH = CfgNode({'INIT_FROM': str(init), 'SUITE': suite['name'], 'INIT_TRAIN_PERCENTAGE': 100})
    reference = ROOT / 'outputs_research/pilot_kvasir_v4/evaluations/dice_corrected/Val/Kvasir/summary.json'
    cfg.RESEARCH.REFERENCE_ADAM20_SUMMARY = str(reference)
    cfg.RESEARCH.REFERENCE_ADAM20_SHA256 = sha256(reference)
    ref40 = ROOT / 'outputs_research/pilot_kvasir_v5/evaluations/adam_budget/Val/Kvasir/summary.json'
    cfg.RESEARCH.REFERENCE_ADAM40_SUMMARY = str(ref40)
    cfg.RESEARCH.REFERENCE_ADAM40_SHA256 = sha256(ref40)
    cfg.seed, cfg.data_percentage, cfg.num_workers = suite['seed'], suite['percentage'], 0
    cfg.output_dir = str(suite_output(suite) / variant)
    cfg.control_dir = str(control_dir(suite))
    cfg.resume = False
    cfg.init_from = None
    return cfg


def plain_config(cfg):
    return json.loads(json.dumps(cfg))


def code_manifest():
    fixed = ['train_v6.py', 'datasets/dataloader.py', 'local_runtime.py', 'utils/main_utils.py',
             'utils/SurfaceDice.py', 'trainers/medclipseg_unimedclip.py', 'trainers/layers.py',
             'trainers/scale_block.py']
    paths = [ROOT / p for p in fixed] + sorted((ROOT / 'research_v6').glob('*.py'))
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
    init_path = getattr(cfg, 'init_from', None) or cfg.RESEARCH.INIT_FROM
    if cfg.data_percentage < cfg.RESEARCH.INIT_TRAIN_PERCENTAGE:
        raise ValueError('This initialization has seen more training data than the requested subset')
    values = {k: plain_config(cfg[k]) for k in ('MODEL', 'TRAIN', 'DATASET', 'TEST', 'RESEARCH')}
    values.update(seed=cfg.seed, percentage=cfg.data_percentage, workers=cfg.num_workers)
    signature = dict(config=values, init_sha256=sha256(init_path),
                     code=code_manifest(), data=data_manifest(cfg),
                     environment={'python': platform.python_version(), 'torch': torch.__version__,
                                  'cuda': torch.version.cuda, 'monai': monai.__version__,
                                  'device': torch.cuda.get_device_name() if cfg.MODEL.DEVICE == 'cuda' else 'cpu'})
    signature['digest'] = object_hash(signature)
    return signature


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
