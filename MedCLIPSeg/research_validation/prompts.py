"""Prespecified interventions; generic never reads image or mask attributes."""
import re
from research_formal.objective import reflect_text

CONDITIONS = ('original', 'no_location', 'generic', 'opposite_location', 'hflip_sync', 'vflip_sync')
# Reuse the sentence fixed in the 2026-09-29 diagnosis, without selecting a new template.
GENERIC = 'a polyp in a colonoscopy image'


def prompt_for(text, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown prompt intervention: '+condition)
    if condition == 'generic':
        return GENERIC
    if condition == 'original':
        return text
    if 'located in ' not in text:
        raise ValueError('The locked source prompts require a location suffix')
    if condition == 'no_location':
        return text[:text.index('located in ')].rstrip(' ,.;')
    if condition == 'hflip_sync':
        return reflect_text(text, -1)
    if condition == 'vflip_sync':
        return reflect_text(text, -2)
    return reflect_text(reflect_text(text, -1), -2)


def direction_bearing(text):
    suffix = text.split('located in ', 1)[-1]
    return bool(re.search(r'\b(left|right|top|bottom)\b', suffix))


def intervention_batch(batch, condition):
    # Forward inference consumes image/text only. GT stays in the original frame.
    result = dict(batch)
    result['text_prompt'] = [prompt_for(t, condition) for t in batch['text_prompt']]
    dim = -1 if condition == 'hflip_sync' else -2 if condition == 'vflip_sync' else None
    if dim is not None:
        result['image'] = batch['image'].flip(dim)
    return result, dim


def align_predictions(draws, dim):
    return draws if dim is None else draws.flip(dim)
