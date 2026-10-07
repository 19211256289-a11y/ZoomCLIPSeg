"""Independent paths and immutable signatures; old formal sources stay untouched."""
import json
from pathlib import Path
import yaml
from research_formal import common as base
from research_formal.common import ROOT, WORKSPACE, CKPT_NAME, now, write_json, sha256, object_hash, plain_config, assert_signature, optimizer_groups

MODES=('generic','mixed')
GENERIC='a polyp in a colonoscopy image'

def load_suite(name='pilot_softcon_kvasir_v1'):
    if name!='pilot_softcon_kvasir_v1':
        raise ValueError('Only the locked source-only pilot is enabled')
    suite=yaml.safe_load((ROOT/'research_softcon/suites'/f'{name}.yaml').read_text(encoding='utf-8'))
    assert suite['name']==name and suite['dataset']=='Kvasir'
    assert suite['modes']==list(MODES) and suite['seeds']==[666]
    assert suite['epochs']==30 and suite['percentage']==100
    assert suite['generic_prompt']==GENERIC and suite['mixed_generic_probability']==.5
    assert suite['evaluation_conditions']==['generic','original'] and suite['evaluation_mc']==30
    assert not suite['test_targets'] and not suite['automatic_test_after_all_training']
    assert suite['checkpoint_selection']=='final_epoch'
    assert suite['softcon_weight']==0.0 and suite['reference_suite']=='pilot_prompt_kvasir_v1'
    suite['runs']=[f'{mode}_seed{s}' for s in suite['seeds'] for mode in MODES]
    return suite

def split_run(run):
    mode,sep,seed=run.rpartition('_seed')
    if not sep or mode not in MODES or seed!='666':raise ValueError(run)
    return mode,int(seed)

def control_dir(suite):return WORKSPACE/'control/research_softcon'/suite['name']
def suite_output(suite):return ROOT/'outputs_research'/suite['name']
def checkpoint_path(suite,run):return suite_output(suite)/run/'Kvasir/trained_models'/f'seed{split_run(run)[1]}'/CKPT_NAME

def make_config(suite,run,checkpointing=True):
    mode,seed=split_run(run)
    assert run in suite['runs']
    cfg=base.make_config(base.load_suite('formal_kvasir_v1'),f'paired_image_seed{seed}',checkpointing)
    cfg.TRAIN.NUM_EPOCHS=suite['epochs']
    cfg.TRAIN.CLIP_WEIGHT=0.0
    cfg.TRAIN.LEARNING_RATE=cfg.TRAIN.NEW_LEARNING_RATE=suite['learning_rate']
    cfg.TRAIN.MIN_LEARNING_RATE=suite['min_learning_rate']
    cfg.TEST.NUM_SAMPLES=suite['evaluation_mc']
    cfg.RESEARCH.SUITE=suite['name']
    cfg.RESEARCH.RUN_ID=run
    cfg.RESEARCH.PROMPT_MODE=mode
    cfg.RESEARCH.GENERIC_PROMPT=GENERIC
    cfg.RESEARCH.GENERIC_PROBABILITY={'attribute':0.,'generic':1.,'mixed':.5}[mode]
    cfg.RESEARCH.VALIDATION_PROMPT='generic'
    cfg.RESEARCH.PROTOCOL_SHA256=sha256(ROOT/'research_softcon/suites'/f'{suite["name"]}.yaml')
    cfg.output_dir=str(suite_output(suite)/run)
    cfg.control_dir=str(control_dir(suite))
    cfg.resume=False
    return cfg

def code_manifest():
    result=base.code_manifest()
    for p in [ROOT/'train_softcon.py',*sorted((ROOT/'research_softcon').rglob('*.py'))]:
        result[p.relative_to(ROOT).as_posix()]=sha256(p)
    return result

def reference_manifest():
    """Reuse only completed, audited, same-budget on-loss controls, never their weights for initialization."""
    old=ROOT/'outputs_research/pilot_prompt_kvasir_v1'
    audit_path=WORKSPACE/'reports/acceptance_prompt_20261003/audit.json'
    audit=read(audit_path)
    if not audit['passed']:raise ValueError('Prompt pilot acceptance required')
    plan=read(old/'experiment_plan.json')
    if plan['digest']!=audit['plan_digest']:raise ValueError('Reference training plan changed')
    for name,h in next(iter(plan['signatures'].values()))['code'].items():
        if sha256(ROOT/name)!=h:raise ValueError('Reference source code changed: '+name)
    results={}
    for mode in MODES:
        run=mode+'_seed666'
        path=old/run/'Kvasir/trained_models/seed666'/CKPT_NAME
        digest=sha256(path)
        if digest!=audit['checks'][run]['checkpoint_sha256']:raise ValueError('Reference checkpoint changed')
        conditions={}
        for condition in ('generic','original'):
            folder=old/'evaluations'/run/condition
            manifest=read(folder/'manifest.json')
            if not manifest['complete']:raise ValueError('Incomplete reference evaluation')
            for name,h in manifest['outputs'].items():
                if sha256(folder/name)!=h:raise ValueError('Reference prediction changed: '+name)
            summary=read(folder/'summary.json')
            if summary!=audit['summaries'][f'{run}/{condition}']:raise ValueError('Reference summary changed')
            conditions[condition]=dict(summary=summary,manifest_sha256=sha256(folder/'manifest.json'),
                                       outputs=manifest['outputs'],folder=str(folder))
        results[run]=dict(checkpoint_sha256=digest,signature=plan['signatures'][run],evaluations=conditions)
    return dict(source_plan_digest=plan['digest'],acceptance_sha256=sha256(audit_path),runs=results,
                role='historical same-seed same-budget controls; never initialization weights')

def make_signature(cfg):
    result=base.make_signature(cfg)
    result['code']=code_manifest()
    result.pop('digest')
    result['digest']=object_hash(result)
    return result

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def lock_plan(suite):
    signatures={r:make_signature(make_config(suite,r)) for r in suite['runs']}
    references=reference_manifest()
    for run,sig in signatures.items():
        old=references['runs'][run]['signature']
        for field in ('data','assets','init_sha256','environment'):
            if sig[field]!=old[field]:raise ValueError('Reference comparability failed: '+field)
        for field in ('MODEL','DATASET','TEST'):
            if sig['config'][field]!=old['config'][field]:raise ValueError('Reference config mismatch: '+field)
        old_train=dict(old['config']['TRAIN']);old_train['CLIP_WEIGHT']=0.0
        if sig['config']['TRAIN']!=old_train:raise ValueError('More than SoftCon weight changed')
    plan=dict(suite=suite,signatures=signatures,checkpointing=True,no_target_tests=True,
              exploration_only=True,primary_condition='generic',primary_comparison='generic without vs with SoftCon',references=references)
    plan['digest']=object_hash(plan)
    path=suite_output(suite)/'experiment_plan.json'
    if path.exists() and read(path)!=plan:raise ValueError('Protocol changed; use a new named suite, never reuse these checkpoints')
    write_json(path,plan)
    for run in suite['runs']:
        cfg=make_config(suite,run)
        path=suite_output(suite)/'configs'/f'{run}.yaml'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump({k:plain_config(cfg[k]) for k in ('DATASET','TRAIN','TEST','MODEL','RESEARCH')},sort_keys=False),encoding='utf-8')
    return plan
