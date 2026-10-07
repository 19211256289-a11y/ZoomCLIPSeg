"""Pausing, resumable source-Val diagnostics for the nine frozen formal models."""
from local_runtime import configure_local_models
configure_local_models()
import argparse
import gc
import json
import os
from pathlib import Path
import time
from statistics import mean, stdev
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_formal import TrainingControl, TrainingPaused, forward_batch, set_random_seed
from research_formal.common import write_json, now, sha256, object_hash
from research_formal.metrics import case_metrics, summarize, write_cases
from research_formal.data import make_dataset
from research_formal.model import build_model
from utils.main_utils import CfgNode, read_text
from .protocol import ROOT, OUT, CONTROL, FORMAL, read, prepare, own_code
from .prompts import CONDITIONS, prompt_for, direction_bearing, intervention_batch, align_predictions


def configuration(run):
    signature = read(FORMAL/'experiment_plan.json')['signatures'][run]
    cfg = CfgNode(signature['config'])
    cfg.data_percentage, cfg.num_workers = cfg.percentage, cfg.workers
    return cfg, signature


def load_model(run, plan):
    cfg, signature = configuration(run)
    saved = torch.load(plan['checkpoints'][run]['path'], map_location='cpu', weights_only=False)
    if saved['research_signature'] != signature or saved['epoch'] != 99 or saved['progress']['phase'] != 'epoch_complete':
        raise ValueError('Only the locked completed formal checkpoint is allowed')
    set_random_seed(cfg.seed)
    model = build_model(cfg)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.eval().to(cfg.MODEL.DEVICE)
    return cfg, model


def infer_batches(model, data, cfg, condition, control, samples=30, batch_size=32):
    """Restarting this iterator reproduces its initial RNG stream and sample order."""
    set_random_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    use_cuda = str(cfg.MODEL.DEVICE).startswith('cuda')
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if control.pause_requested():
                control.update('paused', reason='Resume restarts the interrupted condition with its original seed.')
                raise TrainingPaused()
            view, dim = intervention_batch(batch, condition)
            if use_cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            draws = torch.sigmoid(forward_batch(model, view, cfg.MODEL.DEVICE, samples))
            draws = align_predictions(draws, dim)
            probability = draws.mean(0)
            variance = draws.var(0, unbiased=False)
            entropy = -(probability*torch.log(probability+1e-8)+(1-probability)*torch.log(1-probability+1e-8))
            if use_cuda:
                torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            control.update('running', condition=condition, step=index+1, total_steps=len(loader))
            yield batch, [x.cpu().numpy() for x in (probability,variance,entropy)], seconds
            del draws


def completed_summary(destination, expected_signature=None):
    manifest_path = destination/'manifest.json'
    if not manifest_path.exists():
        return None
    manifest = read(manifest_path)
    if expected_signature is not None and manifest['signature'] != expected_signature:
        raise ValueError('Inputs changed for cached condition: '+str(destination))
    if not manifest.get('complete'):
        return None
    # Artifact checks prevent silently reusing edited or incomplete cached predictions.
    for name, digest in manifest['outputs'].items():
        if sha256(destination/name) != digest:
            raise ValueError('Diagnostic output changed: '+str(destination/name))
    return read(destination/'summary.json')


def evaluate_condition(run, condition, cfg, model, plan, data):
    destination = OUT/'evaluations'/run/condition
    signature = dict(plan_digest=plan['digest'], run=run, condition=condition,
                     checkpoint_sha256=plan['checkpoints'][run]['sha256'])
    done = completed_summary(destination, signature)
    if done is not None:
        return done
    control = TrainingControl(CONTROL, run_id=run, condition=condition, phase='evaluation', step=0)
    if control.pause_requested():
        raise TrainingPaused()
    control.update('starting')
    for sub in ('masks','probabilities'):
        (destination/sub).mkdir(parents=True,exist_ok=True)
    manifest = dict(signature=signature, complete=False, started_at=now())
    write_json(destination/'manifest.json',manifest)
    original = FORMAL/'evaluations'/run/'Val/Kvasir'
    import csv
    with (original/'cases.csv').open(encoding='utf-8-sig',newline='') as stream:
        references = {r['name']:r for r in csv.DictReader(stream)}
    threshold = read(original/'summary.json')['source_train_area_threshold']
    rows, elapsed, artifacts = [], 0., {}
    torch.cuda.reset_peak_memory_stats()
    for batch, (probabilities, variances, entropies), seconds in infer_batches(model,data,cfg,condition,control):
        elapsed += seconds
        for b, name in enumerate(batch['mask_name']):
            probability = probabilities[b]
            assert probability.shape == (224,224) and np.isfinite(probability).all()
            ref = references[name]
            with np.load(original/ref['probability_file']) as archive:
                base = archive['mean'].copy()
            changed = prompt_for(batch['text_prompt'][b], condition) != batch['text_prompt'][b]
            if condition == 'original' or (condition == 'opposite_location' and not changed):
                if not np.allclose(probability,base,atol=1e-5,rtol=1e-5) or not np.array_equal(probability>.5,base>.5):
                    raise ValueError(f'Original-stream replay mismatch for {run}/{name}; do not interpret diagnostics until resolved.')
            index = len(rows)
            png, npz = f'masks/{index:06d}.png', f'probabilities/{index:06d}.npz'
            if not cv2.imwrite(str(destination/png), (probability>.5).astype(np.uint8)*255):
                raise OSError('Could not save PNG')
            np.savez_compressed(destination/npz,mean=probability,variance=variances[b],entropy=entropies[b])
            artifacts[png],artifacts[npz] = sha256(destination/png),sha256(destination/npz)
            row = dict(name=name,image_name=batch['image_name'][b],mask_file=png,probability_file=npz,
                       condition=condition,prompt_changed=changed,direction_bearing=direction_bearing(batch['text_prompt'][b]),
                       original_prompt=batch['text_prompt'][b],input_prompt=prompt_for(batch['text_prompt'][b],condition),
                       original_DSC=float(ref['DSC']),aligned_probability_mae=float(abs(probability-base).mean()),
                       binary_disagreement_fraction=float(((probability>.5)!=(base>.5)).mean()),
                       **case_metrics(Path(cfg.DATASET.VAL_PATH)/'label'/name,probability))
            row['delta_DSC_pp'] = 100*(row['DSC']-row['original_DSC'])
            rows.append(row)
        control.update('running',cases_completed=len(rows),inference_seconds=elapsed)
    if len(rows)!=100 or {r['name'] for r in rows} != set(references):
        raise ValueError('Incomplete source-Val coverage')
    result = summarize(rows,threshold)
    subsets = {}
    for label, selected in [('all',rows),('direction_bearing',[r for r in rows if r['direction_bearing']]),
                            ('prompt_changed',[r for r in rows if r['prompt_changed']])]:
        subsets[label] = dict(n=len(selected), mean_delta_DSC_pp=float(np.mean([r['delta_DSC_pp'] for r in selected])) if selected else None,
                             aligned_probability_mae=float(np.mean([r['aligned_probability_mae'] for r in selected])) if selected else None,
                             binary_disagreement_fraction=float(np.mean([r['binary_disagreement_fraction'] for r in selected])) if selected else None)
    result.update(run_id=run,variant=cfg.MODEL.VARIANT,seed=cfg.seed,condition=condition,
                  mc_samples=30,batch_size=32,split='Val',target='Kvasir',diagnostic=True,
                  prompt_changed_count=sum(r['prompt_changed'] for r in rows),subsets=subsets,
                  inference_seconds=elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  completed_at=now(),checkpoint_sha256=signature['checkpoint_sha256'])
    write_cases(destination/'cases.csv',rows)
    write_json(destination/'summary.json',result)
    artifacts['cases.csv'],artifacts['summary.json'] = sha256(destination/'cases.csv'),sha256(destination/'summary.json')
    manifest.update(complete=True,finished_at=now(),n=100,outputs=artifacts)
    write_json(destination/'manifest.json',manifest)
    control.update('completed',cases_completed=100)
    return result


def report():
    path = OUT/'protocol.json'
    if not path.exists():
        raise ValueError('Prepare the diagnostic protocol first')
    plan = read(path)
    results = {}
    for run in plan['runs']:
        for condition in CONDITIONS:
            s = completed_summary(OUT/'evaluations'/run/condition,
                                  dict(plan_digest=plan['digest'],run=run,condition=condition,
                                       checkpoint_sha256=plan['checkpoints'][run]['sha256']))
            if s:
                results[f'{run}/{condition}'] = s
    aggregate,interventions = {},{}
    for condition in CONDITIONS:
        aggregate[condition] = {}
        interventions[condition] = {}
        for variant in ('paired_repeat','paired_image','paired_aug'):
            values = [results.get(f'{variant}_seed{seed}/{condition}') for seed in (666,667,668)]
            if all(values):
                aggregate[condition][variant] = {metric:dict(mean=mean(v[metric] for v in values),sample_std=stdev(v[metric] for v in values))
                    for metric in ('DSC_percent','small_DSC_percent','NSD_percent','micro_foreground_precision_percent','micro_foreground_recall_percent')}
                interventions[condition][variant]={}
                for subset in ('all','direction_bearing','prompt_changed'):
                    counts=[v['subsets'][subset]['n'] for v in values]
                    if len(set(counts))!=1:
                        raise ValueError('Intervention subset changed across seeds')
                    interventions[condition][variant][subset]=dict(n=counts[0],**{
                        metric:dict(mean=mean(v['subsets'][subset][metric] for v in values),
                                    sample_std=stdev(v['subsets'][subset][metric] for v in values)) if counts[0] else None
                        for metric in ('mean_delta_DSC_pp','aligned_probability_mae','binary_disagreement_fraction')})
    contrasts = {}
    for condition, arms in aggregate.items():
        if all(v in arms for v in ('paired_aug','paired_image')):
            contrasts[condition] = {k:arms['paired_aug'][k]['mean']-arms['paired_image'][k]['mean'] for k in arms['paired_aug']}
    differences = {c:{k:v-contrasts['original'][k] for k,v in scores.items()} for c,scores in contrasts.items() if 'original' in contrasts}
    machine = dict(generated_at=now(),completed_jobs=len(results),total_jobs=54,complete=len(results)==54,
                   diagnostic_only=True,independent_confirmation=False,plan_digest=plan['digest'],
                   summaries=results,aggregates=aggregate,intervention_diagnostics=interventions,aug_minus_image_pp=contrasts,
                   change_in_aug_advantage_vs_original_pp=differences)
    write_json(OUT/'diagnostic_results.json',machine)
    lines = ['# 固定权重提示与空间诊断','',f'已完成 {len(results)}/54。仅源域验证集；属于事后机制诊断，不是独立测试或新方法性能。','',
             'generic 仅改变推理输入，权重仍曾使用属性文本训练；不能把它当作重新训练过的公平无属性基线。no_location 仍含其他属性。镜像输出已反变换，不与原预测做 TTA。','',
             '|条件|训练方案|总体 DSC % 均值 ± 标准差|小目标 DSC %|','|---|---|---:|---:|']
    for condition,arms in aggregate.items():
        for variant,metrics in arms.items():
            a,b=metrics['DSC_percent'],metrics['small_DSC_percent']
            lines.append(f"|{condition}|{variant}|{a['mean']:.4f} ± {a['sample_std']:.4f}|{b['mean']:.4f} ± {b['sample_std']:.4f}|")
    lines += ['','|条件|同步文本相对图像增强的 DSC 变化 pp|该优势相对原提示变化 pp|','|---|---:|---:|']
    for condition,scores in contrasts.items():
        diff=differences.get(condition,{}).get('DSC_percent')
        text='待完成原提示对照' if diff is None else f'{diff:+.4f}'
        lines.append(f"|{condition}|{scores['DSC_percent']:+.4f}|{text}|")
    lines += ['','|条件|方案|有效改写数|全体概率 MAE|全体二值不一致率|方向词子集 DSC 变化 pp|','|---|---|---:|---:|---:|---:|']
    for condition,arms in interventions.items():
        for variant,subsets in arms.items():
            a,b=subsets['all'],subsets['direction_bearing']
            lines.append(f"|{condition}|{variant}|{subsets['prompt_changed']['n']}|{a['aligned_probability_mae']['mean']:.6f}|{a['binary_disagreement_fraction']['mean']:.6f}|{b['mean_delta_DSC_pp']['mean']:+.4f}|")
    lines += ['', '逐图结果包含有效文本改写比例、含方向词子集、概率 MAE 和二值预测不一致率。opposite_location 对 center-only 文本无改写，其输出必须与原提示一致。',
              '本诊断不会自动触发新训练，不会修改正式 Test 主表，也不从这些条件中挑选测试提示。']
    (OUT/'diagnostic_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return machine


def smoke_check(plan):
    import unittest
    from . import tests
    checked = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    if not checked.wasSuccessful():
        raise RuntimeError('Diagnostic unit checks failed')
    run=plan['runs'][0]
    cfg,model=load_model(run,plan)
    records=sorted(read_text(Path(cfg.DATASET.TEXT_PROMPT_PATH)/'Val_text.xlsx'),key=lambda r:r['Image'])[:2]
    data=make_dataset(cfg,'Val',records)
    control=TrainingControl(CONTROL/'smoke',phase='smoke')
    outputs={}
    for condition in CONDITIONS:
        batches=list(infer_batches(model,data,cfg,condition,control,samples=2,batch_size=2))
        value=batches[0][1][0]
        assert value.shape==(2,224,224) and np.isfinite(value).all()
        outputs[condition]=value
    # Native path equivalence: repeat the same dataloader iteration RNG consumption.
    set_random_seed(cfg.seed)
    native_batch=next(iter(DataLoader(data,batch_size=2,shuffle=False,num_workers=0)))
    with torch.no_grad():
        native=torch.sigmoid(forward_batch(model,native_batch,cfg.MODEL.DEVICE,2)).mean(0).cpu().numpy()
    assert np.array_equal(native,outputs['original']), 'Original inference path changed'
    result=dict(passed=True,checked_at=now(),plan_digest=plan['digest'],code=own_code(),unit_tests=checked.testsRun,
                smoke_only=True,scientific_result=False,run=run,cases=2,mc_samples=2,conditions=list(CONDITIONS),
                original_path_bitwise_equal=True,device=torch.cuda.get_device_name(),no_training=True)
    write_json(CONTROL/'check.json',result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print('CHECK PASSED: unit tests, six GPU inference conditions, native-path equivalence. No training.',flush=True)
    return result


def work(action):
    from local_reproduce import acquire_gpu_queue
    with acquire_gpu_queue():
        plan=prepare()
        if action=='check':
            smoke_check(plan)
            return 0
        check_path=CONTROL/'check.json'
        if not check_path.exists() or read(check_path).get('plan_digest')!=plan['digest'] or not read(check_path).get('passed'):
            raise ValueError('Run ResearchValidation.cmd -Action Check before inference')
        queue_path=CONTROL/'queue.json'
        previous=read(queue_path) if queue_path.exists() else {}
        if previous and previous.get('plan_digest')!=plan['digest']:
            raise ValueError('Queue does not match the diagnostic plan')
        if action=='resume':
            if previous.get('state') not in ('paused','failed','running'):
                raise ValueError('No interrupted diagnostic queue to resume')
            (CONTROL/'PAUSE.request').unlink(missing_ok=True)
        elif previous.get('state') in ('paused','failed','running'):
            raise ValueError('Use Resume for the interrupted diagnostic queue')
        queue=previous or dict(jobs={},plan_digest=plan['digest'])
        queue.update(state='running',pid=os.getpid(),updated_at=now(),error=None)
        write_json(queue_path,queue)
        try:
            for run in plan['runs']:
                cfg,model=None,None
                try:
                    for condition in CONDITIONS:
                        if (CONTROL/'PAUSE.request').exists():
                            raise TrainingPaused()
                        key=f'{run}/{condition}'
                        signature=dict(plan_digest=plan['digest'],run=run,condition=condition,checkpoint_sha256=plan['checkpoints'][run]['sha256'])
                        cached=completed_summary(OUT/'evaluations'/run/condition,signature)
                        if cached:
                            queue['jobs'].setdefault(key,{}).update(status='completed')
                            write_json(queue_path,queue)
                            continue
                        queue['current_job']=key
                        job=queue['jobs'].setdefault(key,{})
                        job.update(status='running',started_at=job.get('started_at',now()))
                        write_json(queue_path,queue)
                        if model is None:
                            cfg,model=load_model(run,plan)
                            records=read_text(Path(cfg.DATASET.TEXT_PROMPT_PATH)/'Val_text.xlsx')
                            data=make_dataset(cfg,'Val',records)
                        started=time.perf_counter()
                        try:
                            evaluate_condition(run,condition,cfg,model,plan,data)
                        finally:
                            job['active_seconds']=job.get('active_seconds',0.)+time.perf_counter()-started
                            write_json(queue_path,queue)
                        job.update(status='completed',completed_at=now())
                        queue['updated_at']=now()
                        write_json(queue_path,queue)
                        print('COMPLETED',key,flush=True)
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
            report()
            queue.update(state='completed',completed_at=now())
            write_json(queue_path,queue)
            return 0
        except (TrainingPaused,KeyboardInterrupt):
            queue.update(state='paused',updated_at=now())
            if queue.get('current_job') and queue['jobs'][queue['current_job']].get('status')!='completed':
                queue['jobs'][queue['current_job']]['status']='paused'
            write_json(queue_path,queue)
            print('PAUSED. Completed conditions preserved; Resume restarts the interrupted condition.',flush=True)
            return 75
        except BaseException as exc:
            queue.update(state='failed',error=f'{type(exc).__name__}: {exc}',updated_at=now())
            if queue.get('current_job') and queue['jobs'][queue['current_job']].get('status')!='completed':
                queue['jobs'][queue['current_job']]['status']='failed'
            write_json(queue_path,queue)
            raise


def status():
    path=CONTROL/'queue.json'
    queue=read(path) if path.exists() else {}
    n=sum(x.get('status')=='completed' for x in queue.get('jobs',{}).values())
    print('Queue:',queue.get('state','not_started'),'| Completed:',f'{n}/54','| Pause:',(CONTROL/'PAUSE.request').exists())
    print('Current:',queue.get('current_job','none'))
    p=CONTROL/'trainer_status.json'
    if p.exists():
        print(json.dumps(read(p),ensure_ascii=False,indent=2))
    seconds=[j['active_seconds'] for j in queue.get('jobs',{}).values() if j.get('status')=='completed' and j.get('active_seconds')]
    if seconds:
        print('Approximate remaining minutes:',round((54-n)*float(np.median(seconds))/60,1),'excluding pauses')
    if queue.get('error'):
        print('ERROR:',queue['error'])
    print('Output:',OUT)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['prepare','check','diagnose','pause','resume','status','report'],default='status')
    args=parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    if args.action=='pause':
        CONTROL.mkdir(parents=True,exist_ok=True)
        (CONTROL/'PAUSE.request').write_text(now(),encoding='utf-8')
        print('Pause requested; exit at the next batch boundary.')
    elif args.action=='status':
        status()
    elif args.action=='report':
        print('Completed diagnostic jobs:',report()['completed_jobs'])
    elif args.action=='prepare':
        print('Prepared',prepare()['digest'],'; no inference or training started.')
    else:
        return work(args.action)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
