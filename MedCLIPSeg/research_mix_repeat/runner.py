"""Independent queue for the matched prompt-training pilot."""
from local_runtime import configure_local_models
configure_local_models()
import argparse,gc,json,os,subprocess,sys,time
from pathlib import Path
from statistics import mean
from .common import *

def maybe(path):return read(path) if path.exists() else {}

def require_check(suite):
    result=maybe(control_dir(suite)/'check.json')
    if not result.get('passed') or result.get('code_digest')!=object_hash(code_manifest()) or result.get('suite_sha256')!=sha256(ROOT/'research_mix_repeat/suites'/f'{suite["name"]}.yaml'):
        raise ValueError('Run ResearchMixRepeat.cmd -Action Check: checks absent or stale')

def verify_checkpoint(suite,run,plan):
    import torch
    path=checkpoint_path(suite,run)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert_signature(saved.get('research_signature'),plan['signatures'][run])
    if saved['epoch']!=suite['epochs']-1 or saved['progress']['phase']!='epoch_complete':raise ValueError('Incomplete final checkpoint '+run)
    check=read(control_dir(suite)/'check.json')
    if saved['initial_fingerprints']!=check['initial_fingerprints'][run]:raise ValueError('Initial parameters differ from checked foundation initialization')
    del saved
    return sha256(path)

def report(suite):
    out=suite_output(suite);results={}
    plan=maybe(out/'experiment_plan.json')
    if not plan:return None
    from .evaluate import cached
    for run in suite['runs']:
        cp=checkpoint_path(suite,run)
        if not cp.exists():continue
        digest=sha256(cp)
        for condition in suite['evaluation_conditions']:
            sig=dict(training_signature=plan['signatures'][run]['digest'],checkpoint_sha256=digest,condition=condition,
                     mc_samples=30,batch_size=32,split='Val',target='Kvasir',threshold=.5)
            r=cached(out/'evaluations'/run/condition,sig)
            if r:results[f'{run}/{condition}']=r
    lines=['# 全图路由的两个新种子复核','',f'完成{len(results)}/8项最终Val评估；seed667/668，每组30轮。',
           '模型保持不变，主比较为有病灶图像中背景误分对应的像素精确率；同时约束召回、DSC及小目标DSC。seed666仅为提出假设的探索性来源，不计入主要门槛。','',
           '|模型|种子|推理提示|DSC %|小目标DSC %|精确率 %|召回 %|','|---|---:|---|---:|---:|---:|---:|']
    for key,r in results.items():
        lines.append(f"|{r['query_mode']}|{r['seed']}|{r['condition']}|{r['DSC_percent']:.4f}|{r['small_DSC_percent']:.4f}|{r['micro_foreground_precision_percent']:.4f}|{r['micro_foreground_recall_percent']:.4f}|")
    gate=None
    if all(f'{run}/generic' in results for run in suite['runs']):
        import csv
        from .gate import replication_gate
        cases={}
        for run in suite['runs']:
            with (out/'evaluations'/run/'generic/cases.csv').open(encoding='utf-8-sig',newline='') as stream:
                cases[run]=list(csv.DictReader(stream))
        gate=replication_gate({run:results[f'{run}/generic'] for run in suite['runs']},cases,suite)
        lines+=['',f'新种子稳定性门槛：{"通过" if gate["passed"] else "未通过"}。这是技术重复与计算投入筛选，不是独立数据确认或显著性检验。','',json.dumps(gate,ensure_ascii=False)]
    write_json(out/'results.json',dict(results=results,complete=len(results)==8,gate=gate,primary_condition='generic'))
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return out/'report.md'

def status(suite):
    queue=maybe(control_dir(suite)/'queue.json');progress=maybe(control_dir(suite)/'trainer_status.json')
    print('Queue:',queue.get('state','not_started'),'| Pause:',(control_dir(suite)/'PAUSE.request').exists())
    print('Completed training:',sum(k.startswith('train/') and j['status']=='completed' for k,j in queue.get('jobs',{}).items()),'/4')
    print('Completed evaluations:',sum(k.startswith('val/') and j['status']=='completed' for k,j in queue.get('jobs',{}).items()),'/8')
    print('Current:',queue.get('current_job','none'));print(json.dumps(progress,ensure_ascii=False,indent=2))
    rates=[j['active_seconds']/suite['epochs'] for k,j in queue.get('jobs',{}).items() if k.startswith('train/') and j['status']=='completed']
    valid=queue.get('current_job')=='train/'+progress.get('run_id','')
    if valid and progress.get('remaining_seconds') and progress.get('epoch'):
        rates.append(progress['remaining_seconds']/max(1,suite['epochs']-progress['epoch']+1))
    rate=mean(rates) if rates else 62.
    remaining=0
    for run in suite['runs']:
        if queue.get('jobs',{}).get('train/'+run,{}).get('status')!='completed':
            epochs=suite['epochs']
            if valid and progress.get('run_id')==run:epochs-=max(0,progress.get('epoch',1)-1)
            remaining+=max(0,epochs)*rate
        for c in suite['evaluation_conditions']:
            if queue.get('jobs',{}).get(f'val/{run}/{c}',{}).get('status')!='completed':remaining+=70
    print('Estimated remaining hours:',round(remaining/3600,2),'(excludes pauses; estimate until actual epochs complete)')
    if queue.get('error'):print('ERROR:',queue['error'])
    print('Output:',suite_output(suite))

def work(suite,action):
    from local_reproduce import acquire_gpu_queue
    from train_mix_repeat import TrainingPaused
    import torch
    directory=control_dir(suite);directory.mkdir(parents=True,exist_ok=True)
    with acquire_gpu_queue():
        if action=='check':
            from .check import run_checks
            run_checks(suite);lock_plan(suite)
            return 0
        require_check(suite);plan=lock_plan(suite)
        if action=='prepare':print('Prepared; no scientific training started.');return 0
        path=directory/'queue.json';q=maybe(path)
        if q.get('plan_digest') and q['plan_digest']!=plan['digest']:raise ValueError('Queue protocol mismatch')
        if action=='resume':
            if q.get('state') not in ('running','paused','failed'):raise ValueError('No interrupted queue')
            (directory/'PAUSE.request').unlink(missing_ok=True)
        elif q.get('state') in ('running','paused','failed'):raise ValueError('Use Resume for interrupted work')
        if (directory/'PAUSE.request').exists():raise ValueError('Pause requested; Resume is needed')
        q=q or dict(jobs={})
        q.update(state='running',pid=os.getpid(),plan_digest=plan['digest'],error=None)
        def save():q['updated_at']=now();write_json(path,q)
        save();child=None
        def job(key,fn):
            if (directory/'PAUSE.request').exists():raise TrainingPaused()
            record=q['jobs'].setdefault(key,{})
            if record.get('status')=='completed':return
            q['current_job']=key;record.update(status='running',started_at=record.get('started_at',now()));save()
            start=time.perf_counter()
            try:fn()
            finally:record['active_seconds']=record.get('active_seconds',0)+time.perf_counter()-start;save()
            record.update(status='completed',completed_at=now());save()
            print('COMPLETE',key,flush=True)
            gc.collect();torch.cuda.empty_cache()
        def train(run):
            nonlocal child
            cfg=make_config(suite,run)
            args=[sys.executable,'-u','train_mix_repeat.py','--config-file',str(suite_output(suite)/'configs'/f'{run}.yaml'),
                  '--output-dir',cfg.output_dir,'--seed',str(cfg.seed),'--data_percentage','100','--num-workers','0',
                  '--control-dir',str(directory),'--gradient-checkpointing']
            if checkpoint_path(suite,run).exists():args.append('--resume')
            logs=suite_output(suite)/'logs';logs.mkdir(exist_ok=True)
            with (logs/f'{run}.log').open('a',encoding='utf-8') as stream:
                child=subprocess.Popen(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                q['child_pid']=child.pid;save();code=child.wait()
            child=None;q.pop('child_pid',None)
            if code==75:raise TrainingPaused()
            if code:raise RuntimeError(f'{run} exited {code}; inspect logs/{run}.log')
            q['jobs']['train/'+run]['checkpoint_sha256']=verify_checkpoint(suite,run,plan)
        try:
            from .evaluate import evaluate
            for run in suite['runs']:
                job('train/'+run,lambda r=run:train(r))
                digest=verify_checkpoint(suite,run,plan)
                if q['jobs']['train/'+run].get('checkpoint_sha256')!=digest:raise ValueError('Completed checkpoint was changed')
                for condition in suite['evaluation_conditions']:
                    # Completed jobs are also validated by the final report cache check.
                    job(f'val/{run}/{condition}',lambda r=run,c=condition:evaluate(make_config(suite,r),checkpoint_path(suite,r),suite_output(suite)/'evaluations'/r/c,directory,c))
                    report(suite)
            q.update(state='completed',completed_at=now());save();print('Report:',report(suite),flush=True)
            return 0
        except TrainingPaused:
            q['state']='paused'
            if q.get('current_job') and q['jobs'][q['current_job']]['status']!='completed':q['jobs'][q['current_job']]['status']='paused'
            save();return 75
        except BaseException as exc:
            if child is not None and child.poll() is None:
                (directory/'PAUSE.request').write_text(now());child.wait()
            q.update(state='failed',error=f'{type(exc).__name__}: {exc}');save();raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['check','prepare','train','resume','pause','status','report'],default='status')
    args=p.parse_args();os.chdir(ROOT);suite=load_suite()
    if args.action=='pause':
        directory=control_dir(suite);directory.mkdir(parents=True,exist_ok=True)
        (directory/'PAUSE.request').write_text(now());print('Pause requested; wait for Queue: paused before powering off.');return 0
    if args.action=='status':status(suite);return 0
    if args.action=='report':print(report(suite));return 0
    return work(suite,args.action)

if __name__=='__main__':raise SystemExit(main())
