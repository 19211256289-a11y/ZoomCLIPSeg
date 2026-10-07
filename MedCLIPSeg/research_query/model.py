"""Within-image region contrast refines the mask query, not the text encoder.

No GT, external prototype memory, extra RNG calls or encoder hooks are used.
The inherited stochastic forward (including its unused first MC pass) is intact.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from open_clip_lib.factory import create_model, load_state_dict
from trainers.medclipseg_unimedclip import CustomCLIP
from research_formal.model import strict_base_state, parameter_fingerprints
from .common import ROOT, MODES


class RegionQuery(nn.Module):
    def __init__(self, channels=512, hidden=64, mode='contrast_query'):
        super().__init__()
        if mode not in ('global_query','contrast_query'):raise ValueError(mode)
        self.mode=mode
        self.query=nn.Sequential(nn.LayerNorm(channels),nn.Linear(channels,hidden),nn.GELU())
        self.region=nn.Sequential(nn.LayerNorm(channels),nn.Linear(channels,hidden),nn.GELU())
        self.fuse=nn.Sequential(nn.Linear(3*hidden,hidden),nn.GELU(),nn.Linear(hidden,channels))
        nn.init.zeros_(self.fuse[-1].weight);nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, query, decoded, coarse_logits):
        # Detached guidance prevents changing probabilities merely to manipulate pooling.
        weights=coarse_logits.detach().sigmoid().flatten(2)
        if self.mode=='global_query':weights=torch.full_like(weights,.5)
        features=decoded.flatten(2)
        def pool(w):
            # If all probabilities saturate at zero, fall back to the global mean.
            # This remains finite without GT or a predicted hard foreground crop.
            total=w.sum(-1,keepdim=True)
            normalized=torch.where(total>1e-6,w/total.clamp_min(1e-6),torch.full_like(w,1/w.shape[-1]))
            return torch.bmm(normalized,features.transpose(1,2)).squeeze(1)
        foreground,background=self.region(pool(weights)),self.region(pool(1-weights))
        return self.fuse(torch.cat((self.query(query),foreground,foreground-background),dim=-1))


class QueryCLIP(CustomCLIP):
    def __init__(self,cfg,clip):
        super().__init__(cfg,clip)
        self.query_mode=cfg.RESEARCH.QUERY_MODE
        if self.query_mode!='baseline':self.region_query=RegionQuery(mode=self.query_mode)

    def compute_seg_logits(self,image_features,text_features,B,H,W):
        if self.query_mode=='baseline':
            return super().compute_seg_logits(image_features,text_features,B,H,W)
        text_features=text_features/text_features.norm(dim=-1,keepdim=True)
        cls=image_features[:,0,:];cls=cls/cls.norm(dim=-1,keepdim=True)
        features=image_features[:,1:,:]
        features=features/features.norm(dim=-1,keepdim=True)
        grid=features.reshape(B,H//self.patch_size,W//self.patch_size,-1).permute(0,3,1,2)
        query=self.mask_head(text_features)
        decoded=self.upscale(grid)
        coarse=torch.einsum('bqc,bchw->bqhw',query.unsqueeze(1),decoded)
        correction=self.region_query(query,decoded,coarse)
        logits=torch.einsum('bqc,bchw->bqhw',(query+correction).unsqueeze(1),decoded)
        return F.interpolate(logits,self.im_size,mode='bilinear',align_corners=False).squeeze(1),cls


def build_model(cfg):
    if cfg.RESEARCH.QUERY_MODE not in MODES:raise ValueError('Unknown query mode')
    if cfg.MODEL.VARIANT!='paired_image' or cfg.MODEL.CLIP_MODEL!='unimedclip' or cfg.MODEL.BACKBONE!='ViT-B/16' or cfg.DATASET.SIZE!=224:
        raise ValueError('Only native UniMedCLIP ViT-B/16 BGR224 is supported')
    base=Path(cfg.RESEARCH.BASE_WEIGHTS).resolve()
    if base!=(ROOT/'checkpoints/unimed_clip_vit_b16.pt').resolve() or not base.is_file():raise ValueError('Foundation weights required')
    clip=create_model('ViT-B-16-quickgelu',pretrained='',precision='fp32',device=cfg.MODEL.DEVICE,
                      force_quick_gelu=True,text_encoder_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
    strict_base_state(clip,load_state_dict(str(base)));clip.eval()
    model=QueryCLIP(cfg,clip.float())
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(('pvl_adapters.','mask_head.','upscale.','region_query.')))
    return model
