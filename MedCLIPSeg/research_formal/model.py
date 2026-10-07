"""Native model, strictly loaded foundation weights, fresh segmentation parameters."""
import hashlib
from pathlib import Path
import torch
from open_clip_lib.factory import create_model, load_state_dict
from trainers.medclipseg_unimedclip import CustomCLIP
from .common import ROOT, VARIANTS


def is_new_parameter(name):
    return False


def strict_base_state(model, state):
    # Older Transformers serialized this deterministic, now nonpersistent buffer.
    permitted = 'text_encoder.transformer.embeddings.position_ids'
    if permitted in state and permitted not in model.state_dict():
        value = state[permitted]
        if value.ndim != 2 or value.shape[0] != 1 or not torch.equal(value.cpu(), torch.arange(value.shape[1]).unsqueeze(0)):
            raise ValueError('Unexpected legacy position_ids buffer')
        state = {k: v for k, v in state.items() if k != permitted}
    model.load_state_dict(state, strict=True)


def build_model(cfg):
    if cfg.MODEL.VARIANT not in VARIANTS:
        raise ValueError('Unknown formal variant')
    if cfg.MODEL.CLIP_MODEL != 'unimedclip' or cfg.MODEL.BACKBONE != 'ViT-B/16' or cfg.DATASET.SIZE != 224:
        raise ValueError('Formal supports native UniMedCLIP ViT-B/16, 224 only')
    base = Path(cfg.RESEARCH.BASE_WEIGHTS).resolve()
    if base != (ROOT/'checkpoints/unimed_clip_vit_b16.pt').resolve() or not base.is_file():
        raise ValueError('Only the locked local UniMedCLIP foundation checkpoint is allowed')
    clip = create_model('ViT-B-16-quickgelu', pretrained='', precision='fp32',
                        device=cfg.MODEL.DEVICE, force_quick_gelu=True,
                        text_encoder_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
    strict_base_state(clip, load_state_dict(str(base)))
    clip.eval()
    model = CustomCLIP(cfg, clip.float())
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(('pvl_adapters.', 'mask_head.', 'upscale.')))
    return model


def parameter_fingerprints(model):
    digests = {key: hashlib.sha256() for key in ('frozen', 'trainable')}
    for name, parameter in model.named_parameters():
        digest = digests['trainable' if parameter.requires_grad else 'frozen']
        value = parameter.detach().cpu().contiguous()
        digest.update(f'{name}|{value.dtype}|{list(value.shape)}'.encode())
        digest.update(value.numpy().tobytes())
    return {key: value.hexdigest() for key, value in digests.items()}
