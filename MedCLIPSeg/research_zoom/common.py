"""Paths and the locked protocol of pilot C."""
from pathlib import Path

import yaml

from research_detail_test.common import (ROOT, WORKSPACE, GENERIC, SMALL, read, maybe, now, write_json, sha256,
                                         object_hash, accepted_runs, checkpoint_path)
from research_detail_test.common import load_protocol as load_test_protocol

NAME = 'pilot_zoom_kvasir_v1'
PROTOCOL = ROOT/'research_zoom/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_zoom'/NAME
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports/pilot_zoom_20261005'
RUNS = ('detail_seed666', 'detail_seed667', 'detail_seed668')


def load_protocol():
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    assert protocol['name'] == NAME and protocol['runs'] == list(RUNS) and protocol['prompt'] == GENERIC
    assert (protocol['mc_samples'], protocol['batch_size'], protocol['threshold']) == (30, 32, .5)
    names = [c['name'] for c in protocol['conditions']]
    assert len(set(names)) == len(names) and 's100' in names
    assert [c['scale'] for c in protocol['conditions'] if c['name'] == 's100'] == [1.0]
    assert set(protocol['gate']['shrink_conditions']) <= set(names)
    configs = [c['name'] for c in protocol['configs']]
    assert len(set(configs)) == len(configs) and 'single' not in configs
    assert any(not c['oracle'] for c in protocol['configs'])
    return protocol


def code_manifest():
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'research_zoom').rglob('*.py'))}
