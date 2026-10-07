"""Identical native network for both paired-view arms; no new parameters."""
import torch
from trainers.medclipseg_unimedclip import CustomCLIP,load_unimedclip_to_device
VARIANTS=('paired_aug','paired_consistency')

def is_new_parameter(name):return False

def build_model(cfg):
    if cfg.MODEL.VARIANT not in VARIANTS:raise ValueError('Unknown V6 variant')
    if cfg.MODEL.CLIP_MODEL!='unimedclip' or cfg.MODEL.BACKBONE!='ViT-B/16' or cfg.DATASET.SIZE!=224:
        raise ValueError('V6 supports native UniMedCLIP ViT-B/16, 224 only')
    model=CustomCLIP(cfg,load_unimedclip_to_device(cfg).float())
    for name,param in model.named_parameters():param.requires_grad_(name.startswith(('pvl_adapters.','mask_head.','upscale.')))
    return model

def initialize_from_checkpoint(model,path):
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'],strict=True)
    return []
