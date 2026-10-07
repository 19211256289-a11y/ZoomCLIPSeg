"""Suites, lineage checks and the two kinds of checkpoint source (accepted Kvasir runs, extension suites)."""
import contextlib
import os
import re
from pathlib import Path

import yaml

from research_formal.common import ROOT, WORKSPACE, now, write_json, sha256, object_hash
from research_detail_test.common import read, maybe

SUITES = ('zoom_busi_v1', 'zoom_ablation_kvasir_v1', 'zoom_kvasir_seeds_v1')
CONTROL_ROOT = WORKSPACE/'control/research_zoom_ext'
TRAINING_QUEUE = WORKSPACE/'control/research_detail_ext/queue.json'


def suite_file(name):
    return ROOT/'research_zoom_ext/suites'/f'{name}.yaml'


def control(name):
    return CONTROL_ROOT/name


def output(name):
    return ROOT/'outputs_research'/name


def report_dir(name):
    return WORKSPACE/'reports'/f'{name}_results'


def split_run(run):
    mode, marker, seed = run.rpartition('_seed')
    if not marker:
        raise ValueError(run)
    return mode, int(seed)


def load_suite(name):
    """Also verifies that the method is exactly the one accepted in the pilot and in the Test confirmation."""
    from research_zoom.common import load_protocol as load_pilot
    if name not in SUITES:
        raise ValueError(name)
    suite = yaml.safe_load(suite_file(name).read_text(encoding='utf-8'))
    assert suite['name'] == name and len(set(suite['arms'])) == len(suite['arms']) and suite['seeds']
    for relative, digest in suite['lineage'].items():
        audit = read(WORKSPACE/relative)
        if sha256(WORKSPACE/relative) != digest or not audit['execution_valid'] or audit['gate_verdict'] != 'pass':
            raise ValueError('Acceptance record missing, changed or not passed: '+relative)
    pilot = load_pilot()
    if suite['zoom'] != pilot['zoom']:
        raise ValueError('Zoom parameters differ from the accepted method')
    names = [c['name'] for c in suite['configs']]
    assert len(set(names)) == len(names) and suite['primary_config'] in names and 'single' not in names
    for config in suite['configs']:
        if config['oracle'] or config['zoom_out']:
            raise ValueError('Only the accepted zoom-in method is allowed')
        if config['name'] == 't15' and config not in pilot['configs']:
            raise ValueError('t15 differs from the accepted configuration')
        if config['target_fraction'] == 'source_calibrated':
            assert suite['source'] != 'kvasir_v1' and suite['source_calibration']['reference_fraction'] == .15
        else:
            assert config['target_fraction'] == .15
    if suite['gate']:
        assert suite['gate']['primary_arm'] in suite['arms']
    suite['runs'] = [f'{arm}_seed{seed}' for seed in suite['seeds'] for arm in suite['arms']]
    return suite


def code_manifest():
    files = sorted((ROOT/'research_zoom_ext').rglob('*.py'))+[ROOT/'research_zoom'/n for n in ('evaluate.py', 'geometry.py')]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}


@contextlib.contextmanager
def posix_paths_on_windows():
    """Checkpoints written on the Linux server contain pickled POSIX path objects."""
    import pathlib
    if os.name != 'nt':
        yield
        return
    original = pathlib.PosixPath
    pathlib.PosixPath = pathlib.PurePosixPath
    try:
        yield
    finally:
        pathlib.PosixPath = original


class Source:
    """Where the checkpoints, the sealed test cohorts and the accepted single-pass results come from."""

    def __init__(self, suite):
        self.suite, self.kind = suite, suite['source']
        if self.kind == 'kvasir_v1':
            from research_detail_test import common as test
            self.test = test
            self.prompt, self.small, self.targets = test.GENERIC, test.SMALL, list(test.TARGETS)
            self.test_list, self.results = test.load_protocol()['test_list'], test.OUTPUT
            self.plan_path = test.OUTPUT/'evaluation_plan.json'
        else:
            from research_detail_ext import common as ext
            self.ext, self.ext_suite = ext, ext.load_suite(self.kind)
            self.prompt, self.targets = self.ext_suite['prompt'], list(self.ext_suite['test_targets'])
            self.test_list, self.results = self.ext_suite['test_list'], ext.output(self.ext_suite)
            self.plan_path = self.results/'experiment_plan.json'
            unknown = [r for r in suite['runs'] if r not in self.ext_suite['runs']]
            if unknown:
                raise ValueError(f'Runs not in the source suite: {unknown}')
        self.ood = self.targets[1:]

    def reference(self, run, target):
        return self.results/'evaluations'/run/target

    def checkpoint(self, run):
        if self.kind == 'kvasir_v1':
            return self.test.checkpoint_path(run)
        return self.ext.checkpoint_path(self.ext_suite, run)

    def ready(self):
        """True once every final checkpoint and every accepted single-pass Test result is on this machine."""
        if not self.plan_path.exists():
            return False
        for run in self.suite['runs']:
            if not self.checkpoint(run).is_file():
                return False
            for target in self.targets:
                if not maybe(self.reference(run, target)/'manifest.json').get('complete'):
                    return False
        return True

    def plan(self):
        return read(self.plan_path)

    def small_threshold(self):
        return self.small if self.kind == 'kvasir_v1' else self.plan()['small_threshold']

    def expected_checkpoints(self):
        """The hash each checkpoint had when its accepted single-pass Test results were produced."""
        if self.kind == 'kvasir_v1':
            accepted = self.test.accepted_runs(self.test.load_protocol())
            return {run: accepted[run]['checkpoint_sha256'] for run in self.suite['runs']}
        result = {}
        for run in self.suite['runs']:
            digests = {read(self.reference(run, t)/'manifest.json')['signature']['checkpoint_sha256'] for t in self.targets}
            if len(digests) != 1:
                raise ValueError(f'{run}: single-pass results come from different checkpoints')
            result[run] = digests.pop()
        return result

    def load_model(self, run, expected):
        if self.kind == 'kvasir_v1':
            from research_detail_test.evaluate import load_model
            return load_model(run, expected)
        from research_detail_ext.evaluate import load_model
        with posix_paths_on_windows():
            return load_model(self.ext_suite, run, expected)

    def train_config(self):
        if self.kind == 'kvasir_v1':
            raise ValueError('No calibration for the reference dataset')
        return self.ext.make_config(self.ext_suite, self.suite['runs'][0])


def lesion_fractions(cfg):
    """Area fractions of the positive Train masks (same definition as the small-target threshold)."""
    import cv2
    import numpy as np
    from utils.main_utils import read_text
    areas = []
    for record in read_text(Path(cfg.DATASET.TEXT_PROMPT_PATH)/'Train_text.xlsx'):
        mask = cv2.imread(str(Path(cfg.DATASET.TRAIN_PATH)/'label'/record['Ground Truth']), 0)
        if mask is None:
            raise ValueError('Unreadable training mask: '+record['Ground Truth'])
        area = float((mask > 127).mean())
        if area:
            areas.append(area)
    return np.array(areas)


def resolve_configs(suite, source):
    """Replaces 'source_calibrated' by a number computed from Train masks only. Returns (configs, calibration)."""
    import numpy as np
    if all(c['target_fraction'] != 'source_calibrated' for c in suite['configs']):
        return [dict(c) for c in suite['configs']], None
    from research_detail_formal import common as formal
    reference = lesion_fractions(formal.make_config(formal.load_suite(), 'baseline_seed666'))
    mine = lesion_fractions(source.train_config())
    q = float(np.mean(reference <= suite['source_calibration']['reference_fraction']))
    target = float(np.quantile(mine, q))
    if not 0.01 <= target <= 0.5:
        raise ValueError(f'Implausible calibrated target fraction: {target}')
    calibration = dict(reference_n=len(reference), reference_quantile_rank=q, source_n=len(mine), target_fraction=target,
                       source_median=float(np.median(mine)), reference_median=float(np.median(reference)))
    return [dict(c, target_fraction=target) if c['target_fraction'] == 'source_calibrated' else dict(c)
            for c in suite['configs']], calibration


def seal_cohorts(source):
    """Checks every test list and file against the seal of the source suite, whatever machine wrote that seal."""
    sealed = {}
    cohorts = source.plan()['test_cohorts']
    for target in source.targets:
        cohort = cohorts[target]
        listing = ROOT/'data'/target/'Prompts_Folder'/source.test_list
        expected = cohort.get('list_sha256') or cohort['prompt_sha256']
        if sha256(listing) != expected:
            raise ValueError(f'{target}: test list differs from the sealed list')
        for path, digest in cohort['files'].items():
            kind, name = re.split(r'[\\/]', path)[-2:]
            if sha256(ROOT/'data'/target/'Test_Folder'/kind/name) != digest:
                raise ValueError(f'{target}: test file differs from the sealed file: {kind}/{name}')
        sealed[target] = dict(n=cohort['n'], list_sha256=expected, files_digest=object_hash(sorted(cohort['files'].values())),
                              groups=list(cohort['image_hash_groups'].values()))
    return sealed
