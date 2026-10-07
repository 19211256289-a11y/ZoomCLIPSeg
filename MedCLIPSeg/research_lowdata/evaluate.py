"""Inspection of the reduced training lists and of the trained checkpoints; model loading."""
import math
from pathlib import Path

from .common import ROOT, checkpoint_path, make_config, split_run, training_list, sha256, object_hash


def inspect_lists(suite):
    """The reduced lists exactly as the trainer will read them. Raises if a list is not what the protocol states.
    CPU only; no model and no test result is involved."""
    from utils.main_utils import read_text
    cfg = make_config(suite, suite['runs'][0])
    prompts = Path(cfg.DATASET.TEXT_PROMPT_PATH)
    full = {r['Image'] for r in read_text(prompts/'Train_text.xlsx')}
    test = {r['Image'] for r in read_text(prompts/suite['test_list'])}
    result = {}
    for fraction in suite['fractions']:
        entry = {}
        for split, key in (('Train', 'training_images'), ('Val', 'validation_images')):
            folder = Path(getattr(cfg.DATASET, split.upper()+'_PATH'))
            listing = prompts/f'{split}_text_{fraction}.xlsx'
            records = read_text(listing)
            names = [r['Image'] for r in records]
            if len(set(names)) != len(names) or len(names) != suite[key][f'p{fraction}']:
                raise ValueError(f'{listing.name}: {len(names)} records, expected {suite[key][f"p{fraction}"]} distinct images')
            if set(names) & test:
                raise ValueError(f'{listing.name} names images of the test list')
            if split == 'Train' and not set(names) <= full:
                raise ValueError(f'{listing.name} is not a subset of the full training list')
            files = {f'{kind}/{r[column]}': sha256(folder/kind/r[column])
                     for r in records for kind, column in (('img', 'Image'), ('label', 'Ground Truth'))}
            entry[split] = dict(n=len(names), list_sha256=sha256(listing), files_digest=object_hash(files))
        steps = math.ceil(entry['Train']['n']/cfg.TRAIN.BATCH_SIZE)
        entry['optimisation_steps'] = dict(per_epoch=steps, total=steps*cfg.TRAIN.NUM_EPOCHS, batch_size=cfg.TRAIN.BATCH_SIZE)
        result[f'p{fraction}'] = entry
    steps = math.ceil(len(full)/cfg.TRAIN.BATCH_SIZE)
    result['full'] = dict(n=len(full), optimisation_steps=dict(per_epoch=steps, total=steps*cfg.TRAIN.NUM_EPOCHS,
                                                               batch_size=cfg.TRAIN.BATCH_SIZE))
    return result


def run_check(suite):
    """Before any training: the lists, the test files and the preprocessing of the second pass."""
    from utils.main_utils import read_text
    import research_zoom.evaluate as zoom_evaluate
    lists = inspect_lists(suite)
    cohorts = {}
    for target in suite['test_targets']:
        folder = ROOT/'data'/target/'Test_Folder'
        records = read_text(ROOT/'data'/target/'Prompts_Folder'/suite['test_list'])
        missing = [r['Image'] for r in records if not (folder/'img'/r['Image']).is_file() or not (folder/'label'/r['Ground Truth']).is_file()]
        if missing:
            raise FileNotFoundError(f'{target}: {len(missing)} test files are missing, e.g. {missing[0]}')
        cohorts[target] = len(records)
    target = suite['test_targets'][0]
    zoom_evaluate.check_preprocessing(ROOT/'data'/target/'Test_Folder',
                                      read_text(ROOT/'data'/target/'Prompts_Folder'/suite['test_list']))
    if not (ROOT/'train_controls.py').is_file():
        raise FileNotFoundError('train_controls.py (entry point of the accepted trainer) is missing')
    return dict(passed=True, lists=lists, cohorts=cohorts)


def inspect_checkpoint(suite, run):
    """The completed final-epoch checkpoint of this run, trained on the reduced list of its fraction."""
    import torch
    arm, fraction, seed = split_run(suite, run)
    cfg = make_config(suite, run)
    path = checkpoint_path(suite, run)
    digest = sha256(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['epoch'] != cfg.TRAIN.NUM_EPOCHS-1 or saved['progress']['phase'] != 'epoch_complete':
        raise ValueError(f'{run}: not the completed final-epoch checkpoint')
    signature = saved['research_signature']
    stored = signature['config']
    if (stored['RESEARCH']['RUN_ID'], stored['RESEARCH']['CONTROL_ARM'], stored['seed'], stored['RESEARCH']['SUITE'],
            stored['percentage'], stored['RESEARCH']['QUERY_MODE']) != (run, arm, seed, suite['name'], fraction,
                                                                         suite['arms'][arm]['query_mode']):
        raise ValueError(f'{run}: checkpoint belongs to another run, suite or training fraction')
    listing = training_list(suite, fraction)
    list_hashes = [value for key, value in signature['data'].items() if Path(key).name == listing.name]
    images = [key for key in signature['data']
              if Path(key).parent.name == 'img' and Path(key).parent.parent == Path(cfg.DATASET.TRAIN_PATH)]
    if list_hashes != [sha256(listing)] or len(images) != suite['training_images'][f'p{fraction}']:
        raise ValueError(f'{run}: the checkpoint was not trained on {listing.name}')
    return saved, cfg, digest


def load_model(suite, run, expected_sha256=None):
    from train_detail_formal import set_random_seed
    from research_detail_formal.model import build_model
    saved, cfg, digest = inspect_checkpoint(suite, run)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f'{run}: final checkpoint changed after training completed')
    set_random_seed(cfg.seed)
    model = build_model(cfg)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.eval().to(cfg.MODEL.DEVICE)
    return model, cfg, digest
