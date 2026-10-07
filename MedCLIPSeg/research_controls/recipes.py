"""The pieces that differ between arms: the training objective of the single-view arms and two dataset variants.

Everything else (model, optimiser, schedule, checkpointing, pause and resume) is the accepted trainer, unchanged.
All random draws use Python's `random`, whose state the trainer saves and restores, so pause and resume stay exact.
"""
import math
import random

SCALE_RANGE = (0.5, 2.0)
TRANSLATION = 0.1
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
BLACK = [-m/s for m, s in zip(MEAN, STD)]     # a black pixel after normalisation


def single_view_objective(model, batch, cfg, ce_loss, dice_loss, forward_batch, calc_loss):
    """One view per optimisation step, as in the original training loop. Optionally an ordinary random flip."""
    import research_detail_formal.objective as accepted
    recipe = cfg.RESEARCH.CONTROL_RECIPE
    view = dict(batch)
    view['text_prompt'] = [accepted.GENERIC]*len(batch['text_prompt'])
    flipped = 0
    if recipe == 'single_flip':
        image, mask = batch['image'].clone(), batch['ground_truth_mask'].clone()
        detail = batch['image_detail'].clone() if 'image_detail' in batch else None
        for index in range(image.shape[0]):
            draw, dim = random.random(), random.choice((-1, -2))   # two draws for every sample, used or not
            if draw < .5:
                image[index] = image[index].flip(dim)
                mask[index] = mask[index].flip(dim)
                if detail is not None:
                    detail[index] = detail[index].flip(dim)
                flipped += 1
        view['image'], view['ground_truth_mask'] = image, mask
        if detail is not None:
            view['image_detail'] = detail
    elif recipe != 'single':
        raise ValueError(recipe)
    logits, softcon = forward_batch(model, view, cfg.MODEL.DEVICE)
    loss = calc_loss(logits, view['ground_truth_mask'].to(cfg.MODEL.DEVICE), ce_loss, dice_loss, cfg)+cfg.TRAIN.CLIP_WEIGHT*softcon
    return loss, dict(prompt_mode='generic', recipe=recipe, flipped=flipped, batch_size=len(view['text_prompt']),
                      supervised_loss=float(loss.detach()), forward_backward_passes=1, dice_mode=cfg.TRAIN.DICE_MODE)


def scale_sample(sample, scale, tx, ty):
    """Applies one scaling and shift to the 224 image, the 448 detail image and the mask."""
    import torch
    from torchvision.transforms import functional as F
    from torchvision.transforms import InterpolationMode
    result = dict(sample)
    for key in ('image', 'image_detail'):
        side = sample[key].shape[-1]
        result[key] = F.affine(sample[key], angle=0., translate=[round(tx*side), round(ty*side)], scale=scale, shear=[0., 0.],
                               interpolation=InterpolationMode.BILINEAR, fill=BLACK)
    mask = sample['ground_truth_mask']
    side = mask.shape[-1]
    moved = F.affine(mask[None].to(torch.float32), angle=0., translate=[round(tx*side), round(ty*side)], scale=scale,
                     shear=[0., 0.], interpolation=InterpolationMode.NEAREST, fill=[0.])
    result['ground_truth_mask'] = moved[0].round().to(mask.dtype)
    return result


class ScaleAugmented:
    """Ordinary random scaling of training samples (after the accepted rotation augmentation)."""

    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        sample = self.inner[index]
        scale = math.exp(random.uniform(math.log(SCALE_RANGE[0]), math.log(SCALE_RANGE[1])))
        tx, ty = random.uniform(-TRANSLATION, TRANSLATION), random.uniform(-TRANSLATION, TRANSLATION)
        return scale_sample(sample, scale, tx, ty)


def make_dataset_factory(data_mode):
    """Replacement for research_detail_formal.data.make_dataset for the two data variants."""
    import research_detail_formal.data as accepted
    from research.data import DualScaleDataset, DualScaleTransform

    def make_dataset(cfg, split, records, condition=None):
        if split not in ('Train', 'Val'):
            raise ValueError('Training does not access Test')
        inner = DualScaleDataset(getattr(cfg.DATASET, split.upper()+'_PATH'), cfg.DATASET.NAME, records,
                                 DualScaleTransform(split == 'Train'), image_size=224, detail_size=448,
                                 upsampled=data_mode == 'upsampled_detail')
        data = accepted.PromptDataset(inner, generic=split == 'Val')
        return ScaleAugmented(data) if data_mode == 'scale_aug' and split == 'Train' else data

    return make_dataset
