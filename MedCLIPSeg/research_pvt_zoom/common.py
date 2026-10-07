"""Protocol and paths of SCZ applied to the Polyp-PVT reference models. Evaluation only."""
import yaml

from research_polyp_pvt import common as pvt
from research_polyp_pvt.common import (ROOT, WORKSPACE, now, write_json, sha256, object_hash, read, maybe, Paused, Control,
                                       TARGETS, OOD, HELD_OUT, COHORT_SIZES)

NAME = 'pvt_zoom_v1'
PROTOCOL = ROOT/'research_pvt_zoom/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_pvt_zoom'
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports'/f'{NAME}_results'
ALL = TARGETS+(HELD_OUT,)


def load_protocol():
    """Also verifies that the zoom is exactly the accepted configuration t15."""
    from research_zoom.common import load_protocol as load_pilot
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    source = pvt.load_protocol()
    pilot = load_pilot()
    accepted = next(c for c in pilot['configs'] if c['name'] == 't15')
    assert protocol['name'] == NAME and protocol['source_suite'] == pvt.NAME and protocol['seeds'] == source['seeds']
    assert protocol['input_size'] == source['recipe']['input_size'] and protocol['threshold'] == .5
    if protocol['zoom'] != pilot['zoom'] or protocol['target_fraction'] != accepted['target_fraction'] \
            or accepted['oracle'] or accepted['zoom_out']:
        raise ValueError('The zoom differs from the accepted method')
    protocol['small_target_threshold'] = source['small_target_threshold']
    return protocol, source


def evaluation_dir(seed, target):
    return OUTPUT/'evaluations'/f'seed{seed}'/target


def job_keys(protocol):
    return ['check']+[f'zoom/seed{seed}/{target}' for seed in protocol['seeds'] for target in ALL]+['report']


def code_manifest():
    files = sorted((ROOT/'research_pvt_zoom').glob('*.py'))
    files += [ROOT/'research_zoom/geometry.py', ROOT/'research_formal/metrics.py']
    files += sorted((ROOT/'research_polyp_pvt/lib').glob('*.py'))+[ROOT/'research_polyp_pvt'/n for n in ('common.py', 'data.py', 'evaluate.py')]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
