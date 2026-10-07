"""MedCLIPSeg with an opt-in full-image detail path and visual-key beta."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from trainers.medclipseg_unimedclip import CustomCLIP, load_unimedclip_to_device, build_medclipseg_unimedclip
from .data import DETAIL_VARIANTS
from .modules import DetailEncoder, DetailDecoder, EvidenceAdapter


VARIANTS = ('legacy_continue', 'dice_corrected', 'detail', 'detail_beta',
            'detail_upsampled', 'detail_scalar_beta')


def fusion_stage(x_img, x_txt, attention_mask, detail_tokens, block, layer, adapter):
    # Explicit arguments keep activation recomputation independent of mutable caches.
    regularization = x_img.new_zeros(())
    betas = x_img.new_empty((x_img.shape[1], 0))
    if isinstance(adapter, EvidenceAdapter):
        visual, textual, regularization, betas = adapter(x_img.transpose(1, 0), x_txt, detail_tokens)
        x_img, x_txt = x_img + visual.transpose(1, 0), x_txt + textual
    elif adapter is not None:
        visual, textual = adapter(x_img.transpose(1, 0), x_txt)
        x_img, x_txt = x_img + visual.transpose(1, 0), x_txt + textual
    return block(x_img), layer(x_txt, attention_mask=attention_mask)[0], regularization, betas


class DetailCLIP(CustomCLIP):
    def __init__(self, cfg, clip_model):
        super().__init__(cfg, clip_model)
        self.detail_encoder = DetailEncoder()
        self.detail_decoder = DetailDecoder(self.text_proj_dim)
        if cfg.MODEL.VARIANT in ('detail_beta', 'detail_scalar_beta'):
            self.pvl_adapters = nn.ModuleList([
                EvidenceAdapter(in_channels_vis=self.embed_dim, in_channels_txt=self.embed_dim,
                                adapter_channels=cfg.MODEL.ADAPTER_DIM, beta=self.beta,
                                gate_init=self.gate_init, scalar=cfg.MODEL.VARIANT == 'detail_scalar_beta')
                for _ in self.fusion_stages])
        self.last_regularization = None
        self.last_beta = None

    def encode_with_detail(self, tokenized, prompts, image, detail_tokens):
        attention_mask = (tokenized != self.text_model.config.pad_token_id).long()
        x_txt = self.text_model.transformer.embeddings(inputs_embeds=prompts)
        extended = attention_mask[:, None, None, :].to(dtype=self.dtype)
        extended = (1.0-extended)*torch.finfo(self.dtype).min
        x_img = self.vision_model.conv1(image)
        x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], -1).permute(0, 2, 1)
        cls = self.vision_model.class_embedding.to(x_img.dtype) + torch.zeros(
            x_img.shape[0], 1, x_img.shape[-1], dtype=x_img.dtype, device=x_img.device)
        x_img = torch.cat([cls, x_img], 1)
        x_img = self.vision_model.ln_pre(x_img + self.vision_model.positional_embedding.to(x_img.dtype))
        x_img = x_img.permute(1, 0, 2)
        regularizers, betas = [], []
        for i, (block, layer) in enumerate(zip(self.vision_model.transformer.resblocks,
                                               self.text_model.transformer.encoder.layer)):
            adapter = self.pvl_adapters[self.fusion_stages.index(i)] if i in self.fusion_stages else None
            args = (x_img, x_txt, extended, detail_tokens, block, layer, adapter)
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                x_img, x_txt, reg, beta = checkpoint(fusion_stage, *args, use_reentrant=False, preserve_rng_state=True)
            else:
                x_img, x_txt, reg, beta = fusion_stage(*args)
            if isinstance(adapter, EvidenceAdapter):
                regularizers.append(reg)
                betas.append(beta)
        x_img = self.vision_model.ln_post(x_img.permute(1, 0, 2))
        if self.vision_model.proj is not None:
            x_img = x_img @ self.vision_model.proj
        text = self.text_model.proj(x_txt[:, 0, :])
        regularization = torch.stack(regularizers).mean() if regularizers else x_img.new_zeros(())
        beta = torch.stack(betas, 1) if betas else None
        return x_img, text, regularization, beta

    def segment(self, features, text, detail, height, width):
        text = text / text.norm(dim=-1, keepdim=True)
        patches = features[:, 1:]
        patches = patches / patches.norm(dim=-1, keepdim=True)
        patches = patches.reshape(features.shape[0], height//self.patch_size,
                                  width//self.patch_size, -1).permute(0, 3, 1, 2)
        decoded = self.upscale(patches)
        original = torch.einsum('bqc,bchw->bqhw', self.mask_head(text).unsqueeze(1), decoded)
        original = F.interpolate(original, self.im_size, mode='bilinear', align_corners=False).squeeze(1)
        return original + self.detail_decoder(decoded, detail)

    def forward(self, image, text, num_samples=30, image_detail=None):
        if image_detail is None or image_detail.shape[-2:] != (448, 448):
            raise ValueError('Detail variants require a full-image 448x448 image_detail tensor')
        b, _, h, w = image.shape
        tokens = self.tokenizer(text).to(image.device)
        with torch.no_grad():
            prompts = self.text_model.transformer.embeddings.word_embeddings(tokens).type(self.dtype)
        detail = self.detail_encoder(image_detail)
        grid = (h//self.patch_size, w//self.patch_size)
        detail_tokens = torch.cat([F.adaptive_avg_pool2d(detail[-1], grid),
                                   F.adaptive_max_pool2d(detail[-1], grid)], 1).flatten(2).transpose(1, 2)
        features, text_features, reg, beta = self.encode_with_detail(tokens, prompts, image, detail_tokens)
        logits = self.segment(features, text_features, detail, h, w)
        self.last_regularization = reg
        self.last_beta = beta.detach() if beta is not None else None
        if self.training:
            patches = features[:, 1:]
            patches = patches / patches.norm(dim=-1, keepdim=True)
            pooled = patches.mean(1)
            i2t = (pooled @ text_features.T) / self.temperature
            t2i = (text_features @ pooled.T) / self.temperature
            with torch.no_grad():
                similarity = (text_features @ text_features.T) / self.temperature
                similarity = similarity / similarity.norm(dim=-1, keepdim=True)
                targets = F.softmax(similarity, dim=-1)
            softcon = (self.soft_cross_entropy(i2t, targets) + self.soft_cross_entropy(t2i, targets.T))/2
            return logits, softcon
        # Keep the author's unused initial forward and exact number/order of random draws.
        samples, beta_sum = [], None
        for _ in range(num_samples):
            features, text_features, _, beta = self.encode_with_detail(tokens, prompts, image, detail_tokens)
            samples.append(self.segment(features, text_features, detail, h, w))
            if beta is not None:
                beta_sum = beta.detach() if beta_sum is None else beta_sum + beta.detach()
        self.last_beta = beta_sum / num_samples if beta_sum is not None else None
        return torch.stack(samples, 0)


def is_new_parameter(name):
    return name.startswith(('detail_encoder.', 'detail_decoder.')) or '.beta_predictor.' in name


def build_model(cfg):
    if cfg.MODEL.VARIANT not in VARIANTS:
        raise ValueError(f'Unknown research variant: {cfg.MODEL.VARIANT}')
    if cfg.MODEL.CLIP_MODEL != 'unimedclip' or cfg.MODEL.BACKBONE != 'ViT-B/16' or cfg.DATASET.SIZE != 224:
        raise ValueError('Research v1 supports UniMedCLIP ViT-B/16 at 224 only')
    if cfg.MODEL.VARIANT not in DETAIL_VARIANTS:
        return build_medclipseg_unimedclip(cfg)
    model = DetailCLIP(cfg, load_unimedclip_to_device(cfg).float())
    for name, param in model.named_parameters():
        param.requires_grad_(is_new_parameter(name) or name.startswith(('pvl_adapters.', 'mask_head.', 'upscale.')))
    return model


def initialize_from_checkpoint(model, path):
    state = torch.load(path, map_location='cpu', weights_only=False)['model']
    expected = model.state_dict()
    unexpected = set(state) - set(expected)
    missing = set(expected) - set(state)
    illegal = {name for name in missing if not is_new_parameter(name)}
    wrong_shape = {name for name in set(state) & set(expected) if state[name].shape != expected[name].shape}
    if unexpected or illegal or wrong_shape:
        raise ValueError(f'Invalid initialization: unexpected={sorted(unexpected)}, missing={sorted(illegal)}, shape={sorted(wrong_shape)}')
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != missing or result.unexpected_keys:
        raise RuntimeError('Checkpoint loading disagrees with the explicit key audit')
    return sorted(missing)
