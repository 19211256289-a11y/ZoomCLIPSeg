"""Paired reflections and label-gated cross-view distillation (training only).

Transformation consistency is established prior art (Bortsova et al., 2019).
This source-only pilot tests image/text alignment; it does not claim novelty.
"""
import random,re
import torch
from torch.nn import functional as F


def reflect_text(text,dim):
    if dim not in (-1,-2):raise ValueError('Only horizontal/vertical reflections supported')
    start=text.find('located in ')
    if start<0:raise ValueError('Unsupported prompt: expected location suffix')
    mapping={'left':'right','right':'left'} if dim==-1 else {'top':'bottom','bottom':'top'}
    return text[:start]+re.sub(r'\b(left|right|top|bottom)\b',lambda m:mapping.get(m[0],m[0]),text[start:])


def reflected_batch(batch,dim):
    result=dict(batch)
    result['image']=batch['image'].flip(dim)
    result['ground_truth_mask']=batch['ground_truth_mask'].flip(dim)
    result['text_prompt']=[reflect_text(t,dim) for t in batch['text_prompt']]
    return result


def cross_view_loss(logits_a,logits_b_aligned,target):
    """Per-pixel teacher is the view closer to the training target, detached.

    Only pixels with a correct teacher at threshold .5 contribute. Foreground and
    background regions are averaged separately per image, then images equally.
    Bernoulli KL via BCEWithLogits retains a useful gradient for saturated wrong
    students. Both-wrong pixels are supervised by the ordinary segmentation loss.
    No GT, teacher selection or second view is used at inference.
    """
    p,q=logits_a.detach().sigmoid(),logits_b_aligned.detach().sigmoid()
    y=target.float();choose_a=(p-y).abs()<=(q-y).abs()
    teacher=torch.where(choose_a,p,q)
    student=torch.where(choose_a,logits_b_aligned,logits_a)
    valid=(teacher>.5)==(y>.5)
    entropy=-(torch.xlogy(teacher,teacher)+torch.xlogy(1-teacher,1-teacher))
    kl=(F.binary_cross_entropy_with_logits(student,teacher,reduction='none')-entropy).clamp_min(0)
    per_image=[]
    for values,mask,allowed in zip(kl,y>.5,valid):
        regions=[]
        for region in (mask,~mask):
            selected=region&allowed
            if selected.any():regions.append(values[selected].mean())
        per_image.append(torch.stack(regions).mean() if regions else values.sum()*0)
    return torch.stack(per_image).mean(),dict(teacher_valid_fraction=float(valid.float().mean()),
        both_wrong_fraction=float((~valid).float().mean()),
        view_probability_MAE=float((p-q).abs().mean()))


def paired_objective(model,batch,cfg,ce_loss,dice_loss,forward_batch,calc_loss):
    dim=random.choice((-1,-2))
    target=batch['ground_truth_mask'].to(cfg.MODEL.DEVICE)
    logits_a,softcon_a=forward_batch(model,batch,cfg.MODEL.DEVICE)
    reflected=reflected_batch(batch,dim)
    logits_b,softcon_b=forward_batch(model,reflected,cfg.MODEL.DEVICE)
    loss_a=calc_loss(logits_a,target,ce_loss,dice_loss,cfg)+cfg.TRAIN.CLIP_WEIGHT*softcon_a
    loss_b=calc_loss(logits_b,target.flip(dim),ce_loss,dice_loss,cfg)+cfg.TRAIN.CLIP_WEIGHT*softcon_b
    consistency,diagnostics=cross_view_loss(logits_a,logits_b.flip(dim),target)
    weight=cfg.TRAIN.CONSISTENCY_WEIGHT if cfg.MODEL.VARIANT=='paired_consistency' else 0.
    supervised=(loss_a+loss_b)/2
    total=supervised+weight*consistency
    diagnostics.update(supervised_paired_loss=float(supervised.detach()),cross_view_KL=float(consistency.detach()),
        consistency_weight=weight,reflection='horizontal' if dim==-1 else 'vertical',forward_backward_passes=2)
    return total,diagnostics
