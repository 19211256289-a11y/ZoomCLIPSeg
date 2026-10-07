"""Protocol, sealed cohort and paths of the one-time ETIS-Larib evaluation."""
import csv
import os
import shutil
from pathlib import Path

import yaml

from research_formal.common import ROOT, WORKSPACE, now, write_json, sha256, object_hash
from research_detail_test.common import read, maybe

NAME = 'zoom_etis_v1'
PROTOCOL = ROOT/'research_zoom_etis/suites'/f'{NAME}.yaml'
CONTROL = WORKSPACE/'control/research_zoom_etis'
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports'/f'{NAME}_results'
COHORT = OUTPUT/'cohort'       # view of the dataset in the img/label layout the accepted evaluation code reads


def load_protocol():
    """Also verifies that the method is exactly the accepted one."""
    from research_zoom.common import load_protocol as load_pilot
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    assert protocol['name'] == NAME and protocol['arms'] == ['baseline', 'detail']
    assert (protocol['mc_samples'], protocol['batch_size'], protocol['threshold']) == (30, 32, .5)
    for relative, digest in protocol['lineage'].items():
        audit = read(WORKSPACE/relative)
        if sha256(WORKSPACE/relative) != digest or not audit['execution_valid'] or audit['gate_verdict'] != 'pass':
            raise ValueError('Acceptance record missing, changed or not passed: '+relative)
    pilot = load_pilot()
    if protocol['zoom'] != pilot['zoom'] or len(protocol['configs']) != 1 or protocol['configs'][0] not in pilot['configs'] \
            or protocol['configs'][0]['name'] != 't15':
        raise ValueError('The zoom configuration differs from the accepted method')
    protocol['seeds'] = [seed for seeds in protocol['sources'].values() for seed in seeds]
    assert protocol['seeds'] == [666, 667, 668, 669, 670]
    protocol['runs'] = [f'{arm}_seed{seed}' for seed in protocol['seeds'] for arm in protocol['arms']]
    return protocol


def source_of(protocol, run):
    seed = int(run.rpartition('_seed')[2])
    return next(name for name, seeds in protocol['sources'].items() if seed in seeds)


def read_manifest(protocol):
    folder = ROOT/protocol['dataset']['folder']
    path = folder/protocol['dataset']['manifest']
    if sha256(path) != protocol['dataset']['manifest_sha256']:
        raise ValueError('The dataset manifest differs from the one sealed in the protocol')
    with open(path, newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != protocol['dataset']['n'] or len({r['image'] for r in rows}) != len(rows) or len({r['mask'] for r in rows}) != len(rows):
        raise ValueError('Unexpected manifest')
    return folder, rows


def prepare_cohort(protocol):
    """Verifies every file against the manifest and exposes it as img/<id>.tif and label/p<id>.tif.

    Files are hard links to the originals (copies if links are not possible); the originals are not touched.
    Returns (folder, records, seal). Computes no performance number."""
    folder, rows = read_manifest(protocol)
    records, files = [], {}
    for sub in ('img', 'label'):
        (COHORT/sub).mkdir(parents=True, exist_ok=True)
    for row in rows:
        for column, digest_column, sub in (('image', 'image_sha256', 'img'), ('mask', 'mask_sha256', 'label')):
            original = folder/row[column]
            if sha256(original) != row[digest_column]:
                raise ValueError('File differs from the manifest: '+row[column])
            view = COHORT/sub/Path(row[column]).name
            if not view.exists():
                try:
                    os.link(original, view)
                except OSError:
                    shutil.copy2(original, view)
            if sha256(view) != row[digest_column]:
                raise ValueError('Cohort view differs from the original: '+str(view))
            files[f'{sub}/{view.name}'] = row[digest_column]
        records.append({'Image': Path(row['image']).name, 'Ground Truth': Path(row['mask']).name, 'Description': protocol['prompt']})
    seal = dict(n=len(records), manifest_sha256=protocol['dataset']['manifest_sha256'], files_digest=object_hash(files))
    return COHORT, records, seal


def overlap_with_known_data(protocol):
    """Exact-duplicate check against the Kvasir Train/Val images and the five polyp test sets used before."""
    _, rows = read_manifest(protocol)
    mine = {r['image_sha256']: r['image'] for r in rows}
    found = {}
    for target in ('Kvasir', 'ColonDB', 'ClinicDB', 'CVC300', 'BKAI'):
        for split in ('Train_Folder', 'Val_Folder', 'Test_Folder'):
            folder = ROOT/'data'/target/split/'img'
            if not folder.is_dir():
                continue
            hits = [f'{p.name} = {mine[sha256(p)]}' for p in sorted(folder.iterdir()) if p.is_file() and sha256(p) in mine]
            if hits:
                found[f'{target}/{split}'] = hits
    return found


def code_manifest():
    files = sorted((ROOT/'research_zoom_etis').rglob('*.py'))+[ROOT/'research_zoom'/n for n in ('evaluate.py', 'geometry.py')]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
