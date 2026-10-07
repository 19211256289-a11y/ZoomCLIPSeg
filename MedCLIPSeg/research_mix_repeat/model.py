"""Reuse the exact previous model; only seeds and the prospective question change."""
from research_spatial.model import build_model as original_build,parameter_fingerprints

def build_model(cfg):
    if cfg.RESEARCH.QUERY_MODE not in ('baseline','global_mix'):raise ValueError('Only the selected two-arm replication is permitted')
    return original_build(cfg)
