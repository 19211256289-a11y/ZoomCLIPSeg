"""Checkpoint loading and the pre-training self-check for the control experiments."""
from pathlib import Path

from .common import checkpoint_path, make_config, split_run, sha256


def inspect_checkpoint(suite, run):
    import torch
    arm, seed = split_run(suite, run)
    cfg = make_config(suite, run)
    path = checkpoint_path(suite, run)
    digest = sha256(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['epoch'] != cfg.TRAIN.NUM_EPOCHS-1 or saved['progress']['phase'] != 'epoch_complete':
        raise ValueError(f'{run}: not the completed final-epoch checkpoint')
    stored = saved['research_signature']['config']
    if (stored['RESEARCH']['RUN_ID'], stored['RESEARCH']['CONTROL_ARM'], stored['seed'], stored['RESEARCH']['SUITE'],
            stored['TRAIN']['DICE_MODE']) != (run, arm, seed, suite['name'], suite['arms'][arm]['dice']):
        raise ValueError(f'{run}: checkpoint belongs to another run or recipe')
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


def upsampled_dataset(folder, records):
    """Evaluation dataset of the detail_up224 arm: the detail input is the 224 image enlarged to 448."""
    from research.data import DualScaleDataset, DualScaleTransform
    from research_detail_test.evaluate import GenericPrompt
    return GenericPrompt(DualScaleDataset(str(folder), 'Val', records, DualScaleTransform(False),
                                          image_size=224, detail_size=448, upsampled=True))


def run_check(suite):
    """CPU-only checks of the new pieces on real training samples, before any training starts."""
    import random
    import torch
    from utils.main_utils import read_text
    from research.data import DualScaleDataset, DualScaleTransform
    from . import recipes
    cfg = make_config(suite, suite['runs'][0])
    records = read_text(Path(cfg.DATASET.TEXT_PROMPT_PATH)/'Train_text.xlsx')[:6]
    data = DualScaleDataset(cfg.DATASET.TRAIN_PATH, cfg.DATASET.NAME, records, DualScaleTransform(False),
                            image_size=224, detail_size=448, upsampled=False)
    sample = data[0]
    same = recipes.scale_sample(sample, 1.0, 0., 0.)
    if not (torch.allclose(same['image'], sample['image'], atol=1e-3) and torch.allclose(same["image_detail"], sample["image_detail"], atol=1e-3)
            and torch.equal(same['ground_truth_mask'], sample['ground_truth_mask'])):
        raise RuntimeError('Scaling by 1 without shift must return the sample unchanged')
    area = float(sample['ground_truth_mask'].float().mean())
    for scale in (0.5, 2.0):
        moved = recipes.scale_sample(sample, scale, 0., 0.)
        if moved['image'].shape != sample['image'].shape or moved['image_detail'].shape != sample['image_detail'].shape \
                or moved['ground_truth_mask'].shape != sample['ground_truth_mask'].shape \
                or moved['ground_truth_mask'].dtype != sample['ground_truth_mask'].dtype \
                or set(moved['ground_truth_mask'].unique().tolist()) - {0, 1}:
            raise RuntimeError('Scaling changed shapes, type or mask values')
        if not torch.isfinite(moved['image']).all():
            raise RuntimeError('Non-finite pixels after scaling')
    small = recipes.scale_sample(sample, 0.5, 0., 0.)
    if area > 0 and not 0.15*area <= float(small['ground_truth_mask'].float().mean()) <= 0.35*area:
        raise RuntimeError('Scaling by 0.5 should reduce the lesion area to about a quarter')
    corner = small['image'][:, 2, 2]
    if not torch.allclose(corner, torch.tensor(recipes.BLACK), atol=1e-4):
        raise RuntimeError('The border of a shrunken image should be black')
    up = DualScaleDataset(cfg.DATASET.TRAIN_PATH, cfg.DATASET.NAME, records, DualScaleTransform(False),
                          image_size=224, detail_size=448, upsampled=True)[0]
    if up['image_detail'].shape != sample['image_detail'].shape or torch.equal(up['image_detail'], sample['image_detail']) \
            or not torch.equal(up['image'], sample['image']):
        raise RuntimeError('The upsampled-detail dataset is not what the control expects')
    # the single-view objective with a stand-in model: number of random draws and tensor handling
    batch = torch.utils.data.default_collate([data[i] for i in range(4)])

    class Config(dict):
        __getattr__ = dict.__getitem__
    stand_in = Config(MODEL=Config(DEVICE='cpu'), TRAIN=Config(CLIP_WEIGHT=.1, DICE_MODE='image'),
                      RESEARCH=Config(CONTROL_RECIPE='single_flip'))
    seen = {}

    def forward(model, view, device):
        seen.update(view)
        return view['image'].mean(1), torch.zeros(())
    random.seed(7)
    state = random.getstate()
    loss, diagnostics = recipes.single_view_objective(None, batch, stand_in, None, None, forward,
                                                      lambda logits, target, *_: (logits-target.float()).pow(2).mean())
    after = random.getstate()
    random.setstate(state)
    for _ in range(4):
        random.random()
        random.choice((-1, -2))
    if random.getstate() != after or not torch.isfinite(loss) or diagnostics['batch_size'] != 4:
        raise RuntimeError('Unexpected behaviour of the random-flip objective')
    if seen['image'].shape != batch['image'].shape or seen['image_detail'].shape != batch['image_detail'].shape \
            or len(set(seen['text_prompt'])) != 1:
        raise RuntimeError('The single-view objective changed the batch layout or the sentence')
    return dict(passed=True, scale_checks=3, flipped_in_check=diagnostics['flipped'])
