"""Lock a separate diagnostic plan without changing any formal run."""
from pathlib import Path
import csv
import json
from research_formal.common import ROOT, WORKSPACE, sha256, object_hash, write_json, now
from .prompts import CONDITIONS, GENERIC

NAME = 'validation_kvasir_v1'
OUT = ROOT/'outputs_research'/NAME
CONTROL = WORKSPACE/'control/research_validation'/NAME
FORMAL = ROOT/'outputs_research/formal_kvasir_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def own_code():
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def validate_formal_inputs():
    plan = read(FORMAL/'experiment_plan.json')
    assert object_hash({k:v for k,v in plan.items() if k!='digest'}) == plan['digest']
    queue = read(WORKSPACE/'control/research_formal/formal_kvasir_v1/queue.json')
    if queue['state'] != 'completed' or sum(k.startswith('test/') and v['status']=='completed' for k,v in queue['jobs'].items()) != 45:
        raise ValueError('Finish the locked formal suite before post-formal diagnostics')
    reference = plan['signatures'][plan['suite']['runs'][0]]
    for run, signature in plan['signatures'].items():
        assert object_hash({k:v for k,v in signature.items() if k!='digest'}) == signature['digest']
        for field in ('code','data','assets','init_sha256'):
            assert signature[field] == reference[field], (run,field)
    for field in ('code','data','assets'):
        for name, digest in reference[field].items():
            p = ROOT/name if field=='code' else Path(name)
            if sha256(p) != digest:
                raise ValueError('Locked formal input changed: '+str(p))
    if sha256(reference['config']['RESEARCH']['BASE_WEIGHTS']) != reference['init_sha256']:
        raise ValueError('Foundation weights changed')
    checkpoints = {}
    originals = {}
    for run in plan['suite']['runs']:
        seed = plan['signatures'][run]['config']['seed']
        path = FORMAL/run/'Kvasir/trained_models'/f'seed{seed}'/'MedCLIPSeg_unimedclip_ViT-B-16_latest.pth'
        folder = FORMAL/'evaluations'/run/'Val/Kvasir'
        summary, manifest = read(folder/'summary.json'), read(folder/'manifest.json')
        digest = sha256(path)
        assert digest == summary['checkpoint_sha256'] == manifest['signature']['checkpoint_sha256']
        assert manifest['complete'] and sha256(folder/'cases.csv') == manifest['cases_sha256']
        assert summary['n']==100 and summary['mc_samples']==30 and summary['split']=='Val'
        assert summary['run_id']==run and summary['seed']==seed and summary['target']=='Kvasir'
        assert manifest['signature']['training_signature']==plan['signatures'][run]['digest']
        assert object_hash(manifest['signature'])==manifest['digest']
        assert summary['source_train_area_threshold']==0.058989688326573426
        assert not summary['diagnostic'] and not summary['subset_check']
        for name,h in manifest['signature']['evaluation_files'].items():
            assert reference['data'][name]==h
        with (folder/'cases.csv').open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==100 and len({r['name'] for r in rows})==100
        # These files are the reference frame for MAE, equivariance and unchanged-prompt checks.
        artifacts={r[k]:sha256(folder/r[k]) for r in rows for k in ('probability_file','mask_file')}
        checkpoints[run] = dict(path=str(path), sha256=digest)
        originals[run] = dict(summary_sha256=sha256(folder/'summary.json'), cases_sha256=manifest['cases_sha256'],artifacts=artifacts)
    return plan, checkpoints, originals


def make_plan():
    formal, checkpoints, originals = validate_formal_inputs()
    result = dict(name=NAME, formal_plan_digest=formal['digest'], checkpoint_selection='frozen_final_epoch100',
                  runs=formal['suite']['runs'], conditions=list(CONDITIONS), jobs=54,
                  target='Kvasir', split='Val', n_cases=100, mc_samples=30, batch_size=32,
                  rng='reset training seed before every condition; common draw order and batch size',
                  generic_prompt=GENERIC, threshold=.5, no_training=True, test_evaluation_allowed=False,
                  purpose='Post-hoc mechanism and input-information audit, not independent confirmation or a new method',
                  no_location_warning='Other per-image attributes remain; this is NOT a GT-independent prompt.',
                  generic_warning='Class-only prompt; trained weights still saw attribute prompts. Not a fair retrained class-only baseline.',
                  reflection_protocol='Inverse-transform probability draws to original frame; do not ensemble with original predictions.',
                  resume_protocol='Keep completed jobs; rerun interrupted condition from its fixed RNG seed.',
                  primary_diagnostic='paired_aug minus paired_image, compared across prompt interventions',
                  secondary_diagnostics=['DSC','small_DSC','NSD','precision','recall','zero_overlap','aligned_probability_MAE','binary_disagreement'],
                  changed_prompt_subset_required=True, no_target_test_selection=True,
                  checkpoints=checkpoints, original_evaluations=originals, code=own_code())
    result['digest'] = object_hash(result)
    return result


def prepare():
    plan = make_plan()
    path = OUT/'protocol.json'
    if path.exists() and read(path)['digest'] != plan['digest']:
        raise ValueError('Diagnostic protocol changed. Preserve old outputs and create a new named suite.')
    write_json(path, plan)
    CONTROL.mkdir(parents=True,exist_ok=True)
    return plan
