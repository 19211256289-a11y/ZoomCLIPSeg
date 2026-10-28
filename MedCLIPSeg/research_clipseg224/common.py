"""Protocol and paths of the CLIPSeg control at 224 x 224 (companion of clipseg_scz_v1)."""
import yaml

from research_polyp_pvt.common import (ROOT, WORKSPACE, now, write_json, sha256, object_hash, read, maybe, Paused, Control)

NAME = 'clipseg224_scz_v1'
COMPANION = 'clipseg_scz_v1'
PROTOCOL = ROOT/'research_clipseg224/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_clipseg224'
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports'/f'{NAME}_results'
CONFIGS = ('single', 'scz')


def model_dir(protocol):
    return ROOT/protocol['model']['directory']


def weight_file(protocol):
    for name in ('model.safetensors', 'pytorch_model.bin'):
        if (model_dir(protocol)/name).is_file():
            return model_dir(protocol)/name
    raise FileNotFoundError(f'CLIPSeg weights not found in {model_dir(protocol)}. Run DownloadCLIPSeg.cmd once (needs the internet).')


def load_protocol():
    """Also verifies that the zoom is the accepted one."""
    from research_zoom.common import load_protocol as load_pilot
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    assert protocol['name'] == NAME and protocol['seeds'] == [666, 667, 668] and list(protocol['groups']) == ['kvasir_full']
    assert protocol['recipe']['input_size'] == 224 and protocol['recipe']['epochs'] == 100 and protocol['companion_suite'] == COMPANION
    pilot = load_pilot()
    accepted = next(c for c in pilot['configs'] if c['name'] == 't15')
    if protocol['zoom'] != pilot['zoom'] or protocol['target_fraction'] != accepted['target_fraction'] \
            or accepted['oracle'] or accepted['zoom_out']:
        raise ValueError('The zoom differs from the accepted method')
    return protocol


def training_dir(group, seed):
    return OUTPUT/'training'/group/f'seed{seed}'


def evaluation_dir(group, seed, target):
    return OUTPUT/'evaluations'/group/f'seed{seed}'/target


def group_protocol(protocol, group, train_n=None):
    """The fields that the accepted data code (research_polyp_pvt.data) reads."""
    spec = protocol['groups'][group]
    return dict(dataset=spec['dataset'], train_list=spec['train_list'], train_n=train_n, recipe=protocol['recipe'],
                seeds=protocol['seeds'])


def job_keys(protocol):
    keys = ['check']
    for group, spec in protocol['groups'].items():
        for seed in protocol['seeds']:
            keys += [f'train/{group}/seed{seed}']+[f'eval/{group}/seed{seed}/{target}' for target in spec['targets']]
    return keys+['report']


def code_manifest():
    files = sorted((ROOT/'research_clipseg224').glob('*.py'))
    files += [ROOT/'research_polyp_pvt/data.py', ROOT/'research_zoom/geometry.py', ROOT/'research_formal/metrics.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
