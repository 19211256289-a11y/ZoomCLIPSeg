"""Independent paths and immutable signatures; old formal sources stay untouched."""
import json
from pathlib import Path
import yaml
from research_formal import common as base
from research_formal.common import ROOT, WORKSPACE, CKPT_NAME, now, write_json, sha256, object_hash, plain_config, assert_signature, optimizer_groups

MODES=('attribute','generic','mixed')
GENERIC='a polyp in a colonoscopy image'

def load_suite(name='pilot_prompt_kvasir_v1'):
    if name!='pilot_prompt_kvasir_v1':
        raise ValueError('Only the locked source-only pilot is enabled')
    suite=yaml.safe_load((ROOT/'research_prompt/suites'/f'{name}.yaml').read_text(encoding='utf-8'))
    assert suite['name']==name and suite['dataset']=='Kvasir'
    assert suite['modes']==list(MODES) and suite['seeds']==[666]
    assert suite['epochs']==30 and suite['percentage']==100
    assert suite['generic_prompt']==GENERIC and suite['mixed_generic_probability']==.5
    assert suite['evaluation_conditions']==['generic','original'] and suite['evaluation_mc']==30
    assert not suite['test_targets'] and not suite['automatic_test_after_all_training']
    assert suite['checkpoint_selection']=='final_epoch'
    suite['runs']=[f'{mode}_seed{s}' for s in suite['seeds'] for mode in MODES]
    return suite

def split_run(run):
    mode,sep,seed=run.rpartition('_seed')
    if not sep or mode not in MODES or seed!='666':raise ValueError(run)
    return mode,int(seed)

def control_dir(suite):return WORKSPACE/'control/research_prompt'/suite['name']
def suite_output(suite):return ROOT/'outputs_research'/suite['name']
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
    cfg.RESEARCH.PROMPT_MODE=mode
    cfg.RESEARCH.GENERIC_PROMPT=GENERIC
    cfg.RESEARCH.GENERIC_PROBABILITY={'attribute':0.,'generic':1.,'mixed':.5}[mode]
    cfg.RESEARCH.VALIDATION_PROMPT='generic'
    cfg.RESEARCH.PROTOCOL_SHA256=sha256(ROOT/'research_prompt/suites'/f'{suite["name"]}.yaml')
    cfg.output_dir=str(suite_output(suite)/run)
    cfg.control_dir=str(control_dir(suite))
    cfg.resume=False
    return cfg

def code_manifest():
    result=base.code_manifest()
    for p in [ROOT/'train_prompt.py',*sorted((ROOT/'research_prompt').rglob('*.py'))]:
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
    signatures={r:make_signature(make_config(suite,r)) for r in suite['runs']}
    plan=dict(suite=suite,signatures=signatures,checkpointing=True,no_target_tests=True,
              exploration_only=True,primary_condition='generic',primary_comparison='mixed minus generic')
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
