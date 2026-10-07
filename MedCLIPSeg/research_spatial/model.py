"""Four conditional query residuals with global or spatially varying routing.

Adaptation of dynamic-kernel ideas, not a replica of CondInst or Mask2Former.
No GT, mask feedback, prototype bank or persistent forward cache is used.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from open_clip_lib.factory import create_model,load_state_dict
from trainers.medclipseg_unimedclip import CustomCLIP
from research_formal.model import strict_base_state,parameter_fingerprints
from .common import ROOT,MODES


class SpatialQuery(nn.Module):
    def __init__(self,channels=512,hidden=64,router_hidden=32,experts=4,mode='spatial_mix'):
        super().__init__()
        if mode not in ('global_mix','spatial_mix'):raise ValueError(mode)
        self.mode,self.count,self.channels=mode,experts,channels
        self.experts=nn.Sequential(nn.LayerNorm(channels),nn.Linear(channels,hidden),nn.GELU(),nn.Linear(hidden,experts*channels))
        self.router=nn.Sequential(nn.Conv2d(channels,router_hidden,1),
                                 nn.GroupNorm(8 if router_hidden%8==0 else 1,router_hidden),nn.GELU(),
                                 nn.Conv2d(router_hidden,router_hidden,3,padding=1,groups=router_hidden),nn.GELU(),
                                 nn.Conv2d(router_hidden,experts,1))
        # Keep a nonuniform router at initialization: the first update can distinguish
        # the experts even though every expert's output is initially zero.
        nn.init.zeros_(self.experts[-1].weight);nn.init.zeros_(self.experts[-1].bias)

    def components(self,query,decoded):
        correction=self.experts(query).reshape(query.shape[0],self.count,self.channels)
        logits=self.router(decoded)
        if self.mode=='global_mix':logits=logits.mean((-2,-1),keepdim=True).expand_as(logits)
        weights=logits.softmax(dim=1)
        masks=torch.einsum('bkc,bchw->bkhw',correction,decoded)
        residual=(weights*masks).sum(dim=1,keepdim=True)
        return residual,weights,correction

    def forward(self,query,decoded):return self.components(query,decoded)[0]


class SpatialCLIP(CustomCLIP):
    def __init__(self,cfg,clip):
        super().__init__(cfg,clip);self.query_mode=cfg.RESEARCH.QUERY_MODE
        if self.query_mode!='baseline':self.spatial_query=SpatialQuery(mode=self.query_mode)

    def compute_seg_logits(self,image_features,text_features,B,H,W):
        if self.query_mode=='baseline':return super().compute_seg_logits(image_features,text_features,B,H,W)
        text_features=text_features/text_features.norm(dim=-1,keepdim=True)
        cls=image_features[:,0,:];cls=cls/cls.norm(dim=-1,keepdim=True)
        features=image_features[:,1:,:];features=features/features.norm(dim=-1,keepdim=True)
        grid=features.reshape(B,H//self.patch_size,W//self.patch_size,-1).permute(0,3,1,2)
        query=self.mask_head(text_features);decoded=self.upscale(grid)
        logits=torch.einsum('bqc,bchw->bqhw',query.unsqueeze(1),decoded)
        logits=logits+self.spatial_query(query,decoded)
        return F.interpolate(logits,self.im_size,mode='bilinear',align_corners=False).squeeze(1),cls


def build_model(cfg):
    if cfg.RESEARCH.QUERY_MODE not in MODES:raise ValueError('Unknown spatial mode')
    if cfg.MODEL.VARIANT!='paired_image' or cfg.MODEL.CLIP_MODEL!='unimedclip' or cfg.MODEL.BACKBONE!='ViT-B/16' or cfg.DATASET.SIZE!=224:
        raise ValueError('Native BGR224 UniMedCLIP required')
    path=Path(cfg.RESEARCH.BASE_WEIGHTS).resolve()
    if path!=(ROOT/'checkpoints/unimed_clip_vit_b16.pt').resolve() or not path.is_file():raise ValueError('Foundation weights required')
    clip=create_model('ViT-B-16-quickgelu',pretrained='',precision='fp32',device=cfg.MODEL.DEVICE,
                      force_quick_gelu=True,text_encoder_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
    strict_base_state(clip,load_state_dict(str(path)));clip.eval()
    model=SpatialCLIP(cfg,clip.float())
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(('pvl_adapters.','mask_head.','upscale.','spatial_query.')))
    return model
