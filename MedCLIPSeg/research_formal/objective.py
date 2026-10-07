"""Equal-view supervised controls; no consistency term and no inference changes."""
import random
import re


def reflect_text(text, dim):
    if dim not in (-1, -2):
        raise ValueError('Only horizontal/vertical reflections are supported')
    start = text.find('located in ')
    if start < 0:
        raise ValueError('Unsupported prompt: expected location suffix')
    mapping = {'left': 'right', 'right': 'left'} if dim == -1 else {'top': 'bottom', 'bottom': 'top'}
    return text[:start] + re.sub(r'\b(left|right|top|bottom)\b',
                                lambda match: mapping.get(match[0], match[0]), text[start:])


def reflected_batch(batch, dim, synchronize_text=True):
    result = dict(batch)
    result['image'] = batch['image'].flip(dim)
    result['ground_truth_mask'] = batch['ground_truth_mask'].flip(dim)
    result['text_prompt'] = ([reflect_text(t, dim) for t in batch['text_prompt']]
                             if synchronize_text else list(batch['text_prompt']))
    return result


def second_view(batch, variant, dim):
    if variant == 'paired_repeat':
        return batch
    if variant not in ('paired_image', 'paired_aug'):
        raise ValueError(f'Unknown paired-view arm: {variant}')
    return reflected_batch(batch, dim, synchronize_text=variant == 'paired_aug')


def paired_objective(model, batch, cfg, ce_loss, dice_loss, forward_batch, calc_loss):
    # Even repeat_view consumes this draw so all arms preserve RNG call order.
    dim = random.choice((-1, -2))
    other = second_view(batch, cfg.MODEL.VARIANT, dim)
    logits_a, softcon_a = forward_batch(model, batch, cfg.MODEL.DEVICE)
    logits_b, softcon_b = forward_batch(model, other, cfg.MODEL.DEVICE)
    target_a = batch['ground_truth_mask'].to(cfg.MODEL.DEVICE)
    target_b = other['ground_truth_mask'].to(cfg.MODEL.DEVICE)
    loss_a = calc_loss(logits_a, target_a, ce_loss, dice_loss, cfg) + cfg.TRAIN.CLIP_WEIGHT * softcon_a
    loss_b = calc_loss(logits_b, target_b, ce_loss, dice_loss, cfg) + cfg.TRAIN.CLIP_WEIGHT * softcon_b
    loss = (loss_a + loss_b) / 2
    return loss, dict(supervised_paired_loss=float(loss.detach()),
                     first_view_loss=float(loss_a.detach()), second_view_loss=float(loss_b.detach()),
                     reflection_draw='horizontal' if dim == -1 else 'vertical',
                     image_reflected=cfg.MODEL.VARIANT != 'paired_repeat',
                     text_synchronized=cfg.MODEL.VARIANT == 'paired_aug',
                     forward_backward_passes=2, consistency_weight=0.0)
