"""Same image augmentation and compute; only training prompt replacement varies."""
import random
from research_formal.objective import reflected_batch
from .common import GENERIC

def prompt_views(batch,cfg):
    dim=random.choice((-1,-2))
    # All arms consume exactly B draws; replacement is shared by both image views.
    draws=[random.random() for _ in batch['text_prompt']]
    probability=cfg.RESEARCH.GENERIC_PROBABILITY
    if probability not in (0.,.5,1.):raise ValueError('Unsupported prompt replacement probability')
    chosen=[v<probability for v in draws]
    first=dict(batch)
    first['text_prompt']=[GENERIC if replace else text for replace,text in zip(chosen,batch['text_prompt'])]
    second=reflected_batch(first,dim,synchronize_text=False)
    return first,second,dict(prompt_mode=cfg.RESEARCH.PROMPT_MODE,generic_count=sum(chosen),batch_size=len(chosen),
                             generic_probability=probability,reflection_draw='horizontal' if dim==-1 else 'vertical')

def paired_objective(model,batch,cfg,ce_loss,dice_loss,forward_batch,calc_loss):
    first,second,diag=prompt_views(batch,cfg)
    za,ca=forward_batch(model,first,cfg.MODEL.DEVICE)
    zb,cb=forward_batch(model,second,cfg.MODEL.DEVICE)
    la=calc_loss(za,first['ground_truth_mask'].to(cfg.MODEL.DEVICE),ce_loss,dice_loss,cfg)+cfg.TRAIN.CLIP_WEIGHT*ca
    lb=calc_loss(zb,second['ground_truth_mask'].to(cfg.MODEL.DEVICE),ce_loss,dice_loss,cfg)+cfg.TRAIN.CLIP_WEIGHT*cb
    loss=(la+lb)/2
    diag.update(supervised_paired_loss=float(loss.detach()),first_view_loss=float(la.detach()),second_view_loss=float(lb.detach()),
                forward_backward_passes=2,consistency_weight=0.,text_synchronized=False)
    return loss,diag
