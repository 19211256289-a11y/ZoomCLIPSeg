"""Independent paths and immutable signatures; old formal sources stay untouched."""
import json,os
from pathlib import Path
import yaml
from research_formal import common as base
from research_formal.common import ROOT, WORKSPACE, CKPT_NAME, now, write_json, sha256, object_hash, plain_config, assert_signature, optimizer_groups

MODES=('baseline','global_mix','detail','detail_global_mix')
GENERIC='a polyp in a colonoscopy image'

def load_suite(name='formal_detail_mix_kvasir_v1'):
    if name!='formal_detail_mix_kvasir_v1':
        raise ValueError('Only the locked source-only pilot is enabled')
    suite=yaml.safe_load((ROOT/'research_detail_formal/suites'/f'{name}.yaml').read_text(encoding='utf-8'))
    assert suite['name']==name and suite['dataset']=='Kvasir'
    assert suite['modes']==list(MODES) and suite['seeds']==[666,667,668]
    assert suite['epochs']==100 and suite['percentage']==100
    assert suite['generic_prompt']==GENERIC and suite['softcon_weight']==.1
    assert suite['evaluation_conditions']==['generic'] and suite['evaluation_mc']==30
    assert not suite['test_targets'] and not suite['automatic_test_after_all_training']
    assert suite['checkpoint_selection']=='final_epoch'
    assert (suite['query_experts'],suite['query_hidden'],suite['router_hidden'])==(4,64,32)
    suite['runs']=[f'{mode}_seed{s}' for s in suite['seeds'] for mode in MODES]
    return suite

def split_run(run):
    mode,sep,seed=run.rpartition('_seed')
    if not sep or mode not in MODES or seed not in ('666','667','668'):raise ValueError(run)
    return mode,int(seed)

def current_host():
    host=os.environ.get('MEDCLIPSEG_HOST','local')
    if host not in ('local','server4090'):raise ValueError('Unknown execution host')
    return host

def assigned_runs(suite):
    seeds=suite['host_assignment'][current_host()]
    return [r for r in suite['runs'] if split_run(r)[1] in seeds]

def control_dir(suite):return WORKSPACE/'control/research_detail_formal'/suite['name']/current_host()
def suite_output(suite):return ROOT/'outputs_research'/suite['name']/current_host()
def checkpoint_path(suite,run):return suite_output(suite)/run/'Kvasir/trained_models'/f'seed{split_run(run)[1]}'/CKPT_NAME

def make_config(suite,run,checkpointing=True):
    mode,seed=split_run(run)
    assert run in suite['runs']
    cfg=base.make_config(base.load_suite('formal_kvasir_v1'),f'paired_image_seed{seed}',checkpointing)
    cfg.TRAIN.NUM_EPOCHS=suite['epochs']
    cfg.TRAIN.LEARNING_RATE=cfg.TRAIN.NEW_LEARNING_RATE=suite['learning_rate']
    cfg.TRAIN.MIN_LEARNING_RATE=suite['min_learning_rate']
    cfg.TEST.NUM_SAMPLES=suite['evaluation_mc']
    cfg.RESEARCH.SUITE=suite['name']
    cfg.RESEARCH.RUN_ID=run
    cfg.RESEARCH.PROMPT_MODE='generic'
    cfg.RESEARCH.QUERY_MODE=mode
    cfg.DATASET.DETAIL_SIZE=448
    cfg.RESEARCH.DETAIL_CHECKPOINTING=True
    cfg.RESEARCH.DETAIL_INITIALIZATION_SEED=seed+448
    cfg.RESEARCH.PARENT_ACCEPTANCE_SHA256=sha256(WORKSPACE/'reports/acceptance_detail_mix_20261004/audit.json')
    cfg.RESEARCH.GENERIC_PROMPT=GENERIC
    cfg.RESEARCH.GENERIC_PROBABILITY=1.
    cfg.TRAIN.CLIP_WEIGHT=.1
    cfg.RESEARCH.VALIDATION_PROMPT='generic'
    cfg.RESEARCH.PROTOCOL_SHA256=sha256(ROOT/'research_detail_formal/suites'/f'{suite["name"]}.yaml')
    cfg.output_dir=str(suite_output(suite)/run)
    cfg.control_dir=str(control_dir(suite))
    cfg.resume=False
    return cfg

def code_manifest():
    from research_detail_mix.common import code_manifest as parent_manifest
    result=parent_manifest()
    for name in ('research/modules.py','research/data.py'):
        result[name]=sha256(ROOT/name)
    for p in [ROOT/'train_detail_formal.py',*sorted((ROOT/'research_detail_formal').rglob('*.py'))]:
        result[p.relative_to(ROOT).as_posix()]=sha256(p)
    return result

def make_signature(cfg):
    result=base.make_signature(cfg)
    result['code']=code_manifest()
    result.pop('digest')
    result['digest']=object_hash(result)
    return result

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def lock_plan(suite):
    audit=read(WORKSPACE/'reports/acceptance_detail_mix_20261004/audit.json')
    if not audit['passed']:raise ValueError('Previous experiment must pass integrity acceptance')
    signatures={r:make_signature(make_config(suite,r)) for r in assigned_runs(suite)}
    plan=dict(suite=suite,signatures=signatures,checkpointing=True,no_target_tests=True,
              execution_host=current_host(),shared_protocol=shared_protocol(suite),
              exploration_only=False,independent_data_confirmation=False,primary_condition='generic',
              primary_comparison='paired seed factorial comparison; final epoch100, no checkpoint selection')
    plan['digest']=object_hash(plan)
    path=suite_output(suite)/'experiment_plan.json'
    if path.exists() and read(path)!=plan:raise ValueError('Protocol changed; use a new named suite, never reuse these checkpoints')
    write_json(path,plan)
    for run in assigned_runs(suite):
        cfg=make_config(suite,run)
        path=suite_output(suite)/'configs'/f'{run}.yaml'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump({k:plain_config(cfg[k]) for k in ('DATASET','TRAIN','TEST','MODEL','RESEARCH')},sort_keys=False),encoding='utf-8')
    return plan


def portable(value):
    if isinstance(value,dict):return {portable(k):portable(v) for k,v in value.items()}
    if isinstance(value,list):return [portable(v) for v in value]
    if isinstance(value,str):
        normalized=value.replace('\\','/')
        prefix=WORKSPACE.as_posix()
        if normalized==prefix or normalized.startswith(prefix+'/'):
            return '${WORKSPACE}'+normalized[len(prefix):]
    return value

def shared_protocol(suite):
    cfg=make_config(suite,suite['runs'][0])
    configs={r:dict(config={k:plain_config(make_config(suite,r)[k]) for k in ('MODEL','TRAIN','DATASET','TEST','RESEARCH')},seed=split_run(r)[1],percentage=100,workers=0) for r in suite['runs']}
    value=portable(dict(suite=suite,configs=configs,code=code_manifest(),data=base.data_manifest(cfg),
                        assets=base.asset_manifest(),init_sha256=sha256(cfg.RESEARCH.BASE_WEIGHTS)))
    value['digest']=object_hash(value)
    return value
