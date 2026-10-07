"""Independent additive high-resolution detail and global-query residuals."""
import copy
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from open_clip_lib.factory import create_model, load_state_dict
from research_spatial.model import SpatialCLIP
from research_formal.model import strict_base_state, parameter_fingerprints
from research.modules import DetailEncoder, DetailDecoder
from .common import ROOT, MODES

class DetailMixCLIP(SpatialCLIP):
    def __init__(self,cfg,clip):
        mode=cfg.RESEARCH.QUERY_MODE;native=copy.deepcopy(cfg)
        native.RESEARCH.QUERY_MODE='global_mix' if mode in ('global_mix','detail_global_mix') else 'baseline'
        super().__init__(native,clip)
        self.combination_mode=mode;self.has_detail=mode in ('detail','detail_global_mix')
        self.detail_checkpointing=cfg.RESEARCH.DETAIL_CHECKPOINTING
        if self.has_detail:
            # Append after native modules to preserve their initialization.
            # Independent CPU RNG stream gives detail-only and combined arms
            # identical new-branch initialization without perturbing native RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(cfg.RESEARCH.DETAIL_INITIALIZATION_SEED)
                self.detail_encoder=DetailEncoder();self.detail_decoder=DetailDecoder(self.text_proj_dim)

    def segment_with_detail(self,image_features,text_features,B,H,W,detail):
        text_features=text_features/text_features.norm(dim=-1,keepdim=True)
        features=image_features[:,1:,:];features=features/features.norm(dim=-1,keepdim=True)
        grid=features.reshape(B,H//self.patch_size,W//self.patch_size,-1).permute(0,3,1,2)
        query=self.mask_head(text_features);decoded=self.upscale(grid)
        logits=torch.einsum('bqc,bchw->bqhw',query.unsqueeze(1),decoded)
        if self.query_mode=='global_mix':logits=logits+self.spatial_query(query,decoded)
        original=F.interpolate(logits,self.im_size,mode='bilinear',align_corners=False).squeeze(1)
        if self.training and self.detail_checkpointing and torch.is_grad_enabled():
            # Both branches are explicit inputs; no mutable feature cache.
            delta=checkpoint(self.detail_decoder,decoded,detail,use_reentrant=False,preserve_rng_state=True)
        else:delta=self.detail_decoder(decoded,detail)
        return original+delta

    def forward(self,image,text,num_samples=30,image_detail=None):
        if not self.has_detail:return super().forward(image,text,num_samples)
        if image_detail is None or image_detail.shape!=(image.shape[0],3,448,448):
            raise ValueError('Detail arms require aligned original-image 448 inputs')
        B,_,H,W=image.shape;tokens=self.tokenizer(text).to(self.device)
        with torch.no_grad():prompts=self.text_model.transformer.embeddings.word_embeddings(tokens).type(self.dtype)
        if self.training and self.detail_checkpointing and torch.is_grad_enabled():
            detail=checkpoint(self.detail_encoder,image_detail,use_reentrant=False,preserve_rng_state=True)
        else:detail=self.detail_encoder(image_detail)
        features,text_features=self.encode_text_image(tokens,prompts,image)
        logits=self.segment_with_detail(features,text_features,B,H,W,detail)
        if self.training:
            patches=features[:,1:,:];patches=patches/patches.norm(dim=-1,keepdim=True);pooled=patches.mean(1)
            i2t=(pooled@text_features.T)/self.temperature;t2i=(text_features@pooled.T)/self.temperature
            with torch.no_grad():
                similarity=(text_features@text_features.T)/self.temperature
                similarity=similarity/similarity.norm(dim=-1,keepdim=True);targets=F.softmax(similarity,dim=-1)
            return logits,(self.soft_cross_entropy(i2t,targets)+self.soft_cross_entropy(t2i,targets.T))/2
        # Preserve native unused first pass and exact MC forward order. Reuse
        # deterministic CNN features only, never stochastic fused predictions.
        samples=[]
        for _ in range(num_samples):
            features,text_features=self.encode_text_image(tokens,prompts,image)
            samples.append(self.segment_with_detail(features,text_features,B,H,W,detail))
        return torch.stack(samples,0)

def build_model(cfg):
    if cfg.RESEARCH.QUERY_MODE not in MODES:raise ValueError('Unknown combination arm')
    if cfg.MODEL.VARIANT!='paired_image' or cfg.MODEL.CLIP_MODEL!='unimedclip' or cfg.MODEL.BACKBONE!='ViT-B/16' or cfg.DATASET.SIZE!=224:
        raise ValueError('Locked native BGR224 UniMedCLIP required')
    if cfg.DATASET.DETAIL_SIZE!=448:raise ValueError('Locked detail size required')
    path=Path(cfg.RESEARCH.BASE_WEIGHTS).resolve()
    if path!=(ROOT/'checkpoints/unimed_clip_vit_b16.pt').resolve() or not path.is_file():raise ValueError('Foundation weights required')
    clip=create_model('ViT-B-16-quickgelu',pretrained='',precision='fp32',device=cfg.MODEL.DEVICE,
        force_quick_gelu=True,text_encoder_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
    strict_base_state(clip,load_state_dict(str(path)));clip.eval();model=DetailMixCLIP(cfg,clip.float())
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(('pvl_adapters.','mask_head.','upscale.','spatial_query.','detail_encoder.','detail_decoder.')))
    return model
