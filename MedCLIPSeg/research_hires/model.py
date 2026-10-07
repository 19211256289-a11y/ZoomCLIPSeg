"""448 output for the accepted model. Native modules and the accepted detail branch are reused unchanged."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from open_clip_lib.factory import create_model, load_state_dict
from research.modules import conv_block
from research_detail_mix.model import DetailMixCLIP
from research_formal.model import strict_base_state
from .common import ROOT, HIRES

MODES = ('sup448', 'detail448')
TRAINABLE = ('pvl_adapters.', 'mask_head.', 'upscale.', 'spatial_query.', 'detail_encoder.', 'detail_decoder.', 'hires_')


class HiResCLIP(DetailMixCLIP):
    """sup448: native logits upsampled to 448. detail448: additionally a 448 residual from the detail decoder."""

    def __init__(self, cfg, clip):
        super().__init__(cfg, clip)
        self.hires_mode = cfg.RESEARCH.HIRES_MODE
        if self.hires_mode not in MODES:
            raise ValueError(self.hires_mode)
        self.im_size = HIRES   # the native path interpolates its 56x56 logits to this size
        self._stem = None
        if self.hires_mode == 'detail448':
            if not self.has_detail:
                raise ValueError('detail448 requires the detail arm')
            # Appended after every accepted module, from a separate CPU stream: the native modules and the
            # accepted detail encoder/decoder start from exactly the same values as in the 224 detail arm.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(cfg.RESEARCH.DETAIL_INITIALIZATION_SEED+1)
                self.hires_stem = conv_block(3, 8)
                self.hires_fuse = conv_block(24, 8)
                self.hires_out = nn.Conv2d(8, 1, 1)
            nn.init.zeros_(self.hires_out.weight)
            nn.init.zeros_(self.hires_out.bias)

    def decode448(self, decoded, fine, middle, coarse, stem):
        d = self.detail_decoder
        x = d.fuse56(torch.cat([d.project(decoded), coarse], 1))
        x = F.interpolate(x, size=middle.shape[-2:], mode='bilinear', align_corners=False)
        x = d.fuse112(torch.cat([x, middle], 1))
        x = F.interpolate(x, size=fine.shape[-2:], mode='bilinear', align_corners=False)
        x = d.fuse224(torch.cat([x, fine], 1))
        x = F.interpolate(x, size=stem.shape[-2:], mode='bilinear', align_corners=False)
        return self.hires_out(self.hires_fuse(torch.cat([x, stem], 1))).squeeze(1)

    def segment_with_detail(self, image_features, text_features, B, H, W, detail):
        if self.hires_mode != 'detail448':
            return super().segment_with_detail(image_features, text_features, B, H, W, detail)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)
        features = image_features[:, 1:, :]
        features = features/features.norm(dim=-1, keepdim=True)
        grid = features.reshape(B, H//self.patch_size, W//self.patch_size, -1).permute(0, 3, 1, 2)
        query = self.mask_head(text_features)
        decoded = self.upscale(grid)
        logits = torch.einsum('bqc,bchw->bqhw', query.unsqueeze(1), decoded)
        if self.query_mode == 'global_mix':
            logits = logits+self.spatial_query(query, decoded)
        original = F.interpolate(logits, self.im_size, mode='bilinear', align_corners=False).squeeze(1)
        fine, middle, coarse = detail
        if self.training and self.detail_checkpointing and torch.is_grad_enabled():
            delta = checkpoint(self.decode448, decoded, fine, middle, coarse, self._stem,
                               use_reentrant=False, preserve_rng_state=True)
        else:
            delta = self.decode448(decoded, fine, middle, coarse, self._stem)
        return original+delta

    def forward(self, image, text, num_samples=30, image_detail=None):
        if self.hires_mode != 'detail448':
            return super().forward(image, text, num_samples, image_detail)
        if image_detail is None or image_detail.shape[-2:] != (HIRES, HIRES):
            raise ValueError('detail448 requires the aligned 448 input')
        # Deterministic; computed once per forward and reused by every MC sample of that forward.
        self._stem = self.hires_stem(image_detail)
        try:
            return super().forward(image, text, num_samples, image_detail)
        finally:
            self._stem = None


def build_model(cfg):
    mode = cfg.RESEARCH.HIRES_MODE
    if mode not in MODES:
        raise ValueError('Use the accepted builder for 224 arms')
    if cfg.MODEL.VARIANT != 'paired_image' or cfg.MODEL.CLIP_MODEL != 'unimedclip' or cfg.MODEL.BACKBONE != 'ViT-B/16' or cfg.DATASET.SIZE != 224:
        raise ValueError('Locked native BGR224 UniMedCLIP required')
    if cfg.DATASET.DETAIL_SIZE != HIRES:
        raise ValueError('Locked detail size required')
    path = Path(cfg.RESEARCH.BASE_WEIGHTS).resolve()
    if path != (ROOT/'checkpoints/unimed_clip_vit_b16.pt').resolve() or not path.is_file():
        raise ValueError('Foundation weights required')
    clip = create_model('ViT-B-16-quickgelu', pretrained='', precision='fp32', device=cfg.MODEL.DEVICE,
                        force_quick_gelu=True, text_encoder_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
    strict_base_state(clip, load_state_dict(str(path)))
    clip.eval()
    model = HiResCLIP(cfg, clip.float())
    for name, parameter in model.named_parameters():
        used = name.startswith(TRAINABLE) and not (mode == 'detail448' and name.startswith('detail_decoder.out.'))
        parameter.requires_grad_(used)   # the 224 output layer is not on the 448 path
    return model


def build_for(cfg):
    """Accepted builder for 224 arms, this module's builder for 448 arms."""
    if cfg.RESEARCH.HIRES_MODE == 'none':
        from research_detail_mix.model import build_model as accepted
        return accepted(cfg)
    return build_model(cfg)
