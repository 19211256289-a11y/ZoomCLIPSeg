"""Protocol and paths of the further Polyp-PVT reference runs (ultrasound; reduced training sets)."""
import yaml

from research_polyp_pvt import common as pvt
from research_polyp_pvt.common import (ROOT, WORKSPACE, now, write_json, sha256, object_hash, read, maybe, Paused, Control,
                                       WEIGHTS)

NAME = 'pvt_ext_v1'
PROTOCOL = ROOT/'research_pvt_ext/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_pvt_ext'
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports'/f'{NAME}_results'


def load_protocol():
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    source = pvt.load_protocol()
    assert protocol['name'] == NAME and protocol['source_protocol'] == pvt.NAME and protocol['seeds'] == source['seeds']
    assert protocol['threshold'] == .5 and list(protocol['groups']) == ['busi_full', 'kvasir_p10', 'kvasir_p25']
    for name, group in protocol['groups'].items():
        assert group['dataset'] == group['targets'][0] and len(group['targets']) == 5, name
    protocol['recipe'] = source['recipe']          # unchanged
    return protocol


def training_dir(group, seed):
    return OUTPUT/'training'/group/f'seed{seed}'


def evaluation_dir(group, seed, target):
    return OUTPUT/'evaluations'/group/f'seed{seed}'/target


def group_protocol(protocol, group, train_n=None):
    """The fields that the accepted Polyp-PVT data and training code read, for one group."""
    spec = protocol['groups'][group]
    return dict(dataset=spec['dataset'], train_list=spec['train_list'], train_n=train_n, recipe=protocol['recipe'],
                seeds=protocol['seeds'])


def accepted_cases(protocol, group, model, seed, target):
    spec = protocol['comparison'][group][model]
    return ROOT/'outputs_research'/spec['suite']/'evaluations'/f"{spec['arm']}_seed{seed}"/target/'cases.csv', spec['config']


def job_keys(protocol):
    keys = ['check']
    for group, spec in protocol['groups'].items():
        for seed in protocol['seeds']:
            keys += [f'train/{group}/seed{seed}']+[f'eval/{group}/seed{seed}/{target}' for target in spec['targets']]
    return keys+['report']


def code_manifest():
    """Code that determines the numbers: the accepted Polyp-PVT package and the wrapper of this suite."""
    package = ROOT/'research_polyp_pvt'
    files = sorted((package/'lib').glob('*.py'))+[package/n for n in ('common.py', 'data.py', 'train.py', 'evaluate.py')]
    files += [ROOT/'research_pvt_ext'/n for n in ('common.py', 'work.py')]+[ROOT/'research_formal/metrics.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
