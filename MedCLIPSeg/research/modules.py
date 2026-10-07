"""Deterministic detail features and bounded visual-key uncertainty penalties."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from trainers.layers import PVL_Adapter


def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                         nn.GroupNorm(8, out_channels), nn.GELU())


class DetailEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(conv_block(a, b, 2), conv_block(b, b))
            for a, b in [(3, 16), (16, 32), (32, 64)]])

    def forward(self, x):
        result = []
        for stage in self.stages:
            x = stage(x)
            result.append(x)
        return result


class DetailDecoder(nn.Module):
    def __init__(self, global_channels=512):
        super().__init__()
        self.project = nn.Conv2d(global_channels, 64, 1)
        self.fuse56 = conv_block(128, 64)
        self.fuse112 = conv_block(96, 32)
        self.fuse224 = conv_block(48, 16)
        self.out = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, global_features, detail):
        fine, middle, coarse = detail
        x = self.fuse56(torch.cat([self.project(global_features), coarse], 1))
        x = F.interpolate(x, size=middle.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse112(torch.cat([x, middle], 1))
        x = F.interpolate(x, size=fine.shape[-2:], mode='bilinear', align_corners=False)
        return self.out(self.fuse224(torch.cat([x, fine], 1))).squeeze(1)


class SpatialBeta(nn.Module):
    def __init__(self, visual_dim=256, base=2.35, scalar=False):
        super().__init__()
        self.base, self.scalar = base, scalar
        if scalar:
            self.raw = nn.Parameter(torch.zeros(()))
        else:
            self.visual = nn.Sequential(nn.Linear(visual_dim, 32), nn.LayerNorm(32))
            self.detail = nn.Sequential(nn.Linear(128, 32), nn.LayerNorm(32))
            self.predict = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.zeros_(self.predict[-1].weight)
            nn.init.zeros_(self.predict[-1].bias)

    def forward(self, visual, detail):
        if self.scalar:
            raw = self.raw.expand(visual.shape[0], visual.shape[1]-1, 1)
        else:
            v, d = self.visual(visual[:, 1:]), self.detail(detail)
            raw = self.predict(torch.cat([v, d, (v-d).abs()], -1))
        patches = self.base * (1 + .5*torch.tanh(raw)).squeeze(-1)
        cls = patches.new_full((patches.shape[0], 1), self.base)
        return torch.cat([cls, patches], 1)


def attention_with_visual_beta(attention, query, context, beta):
    """The original attention operation, with beta broadcast over visual keys."""
    channels = query.shape[-1]
    q = attention.q_proj(query)
    k_mu, k_logvar = attention.k_proj(context).split(channels, -1)
    v_mu, v_logvar = attention.v_proj(context).split(channels, -1)
    k_mu, v_mu = attention.norm_k(k_mu), attention.norm_v(v_mu)
    k_var = F.softplus(k_logvar) + attention.eps
    v_var = F.softplus(v_logvar) + attention.eps
    scores = (q @ k_mu.transpose(1, 2)) / math.sqrt(channels)
    penalty = (q.pow(2) @ k_var.transpose(1, 2)) / channels
    weights = F.softmax(scores - beta[:, None, :]*penalty.sqrt(), dim=-1)
    sample = v_mu + v_var.sqrt()*torch.randn_like(v_var)
    projected = attention.out_proj(weights @ sample)
    gate = torch.sigmoid(attention.gate)
    return gate*projected + (1-gate)*query


class EvidenceAdapter(PVL_Adapter):
    def __init__(self, *args, scalar=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_predictor = SpatialBeta(kwargs['adapter_channels'], kwargs['beta'], scalar)

    def forward(self, vis, text, detail):
        v, t = self.proj_vis_down(vis), self.proj_txt_down(text)
        v_fused = self.two_way.cross_attn_img_to_txt(v, t)
        beta = self.beta_predictor(v_fused, detail)
        t_fused = attention_with_visual_beta(self.two_way.cross_attn_txt_to_img, t, v_fused, beta)
        penalty = ((beta[:, 1:] / self.beta_predictor.base - 1)**2).mean()
        return self.proj_vis_up(v_fused), self.proj_txt_up(t_fused), penalty, beta[:, 1:]
