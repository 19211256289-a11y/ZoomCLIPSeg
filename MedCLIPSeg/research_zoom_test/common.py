"""Paths and the locked protocol of the two-pass Test confirmation."""
import yaml

from research_detail_test.common import (ROOT, WORKSPACE, GENERIC, SMALL, TARGETS, read, maybe, now, write_json, sha256,
                                         object_hash, accepted_runs, split_run)
from research_detail_test.common import load_protocol as load_test_protocol, OUTPUT as ACCEPTED_TEST

NAME = 'test_zoom_kvasir_v1'
PROTOCOL = ROOT/'research_zoom_test/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_zoom_test'/NAME
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports/test_zoom_20261005'
RUNS = ('detail_seed666', 'detail_seed667', 'detail_seed668')
OOD = ('ColonDB', 'ClinicDB', 'CVC300', 'BKAI')


def load_protocol():
    """Also verifies that the configuration is exactly the one selected and accepted in the pilot."""
    from research_zoom.common import load_protocol as load_pilot, REPORT as PILOT_REPORT
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    assert protocol['name'] == NAME and protocol['runs'] == list(RUNS) and protocol['prompt'] == GENERIC
    assert protocol['test_targets'] == list(TARGETS) and protocol['ood_targets'] == list(OOD)
    assert (protocol['mc_samples'], protocol['batch_size'], protocol['threshold']) == (30, 32, .5)
    audit_path = WORKSPACE/protocol['pilot_acceptance_audit']
    if sha256(audit_path) != protocol['pilot_acceptance_sha256']:
        raise ValueError('Pilot acceptance record changed since this protocol was locked')
    audit, pilot = read(audit_path), load_pilot()
    if not audit['execution_valid'] or audit['gate_verdict'] != 'pass' or audit['selected'] != protocol['config']['name']:
        raise ValueError('The pilot acceptance does not support this configuration')
    if sha256(PILOT_REPORT/'results.json') != audit['results_sha256']:
        raise ValueError('Pilot results changed after acceptance')
    if protocol['zoom'] != pilot['zoom'] or protocol['config'] not in pilot['configs'] or protocol['config']['oracle']:
        raise ValueError('Configuration differs from the pilot')
    return protocol


def code_manifest():
    files = sorted((ROOT/'research_zoom_test').rglob('*.py'))+[ROOT/'research_zoom'/n for n in ('evaluate.py', 'geometry.py')]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
