"""Read-only CPU acceptance of all 54 frozen-model source-Val diagnostics."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import csv
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ROOT = WORKSPACE/'MedCLIPSeg'
sys.path.insert(0, str(ROOT))
from research_validation.protocol import make_plan, read, OUT, CONTROL, FORMAL
from research_validation.prompts import CONDITIONS, prompt_for, direction_bearing
from research_formal.common import sha256, object_hash
from research_formal.metrics import case_metrics, summarize
from utils.main_utils import read_text
cv2.setNumThreads(1)
torch.set_num_threads(1)

def close(a, b, key=''):
    if a is None:
        assert b is None or b == '', (key, a, b)
    else:
        assert np.allclose(a, np.asarray(b, dtype=float), atol=1e-8, rtol=1e-6), (key, a, b)

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def write_json(name, obj):
    (HERE/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def moments(x):
    return dict(mean=float(np.mean(x)), sample_sd=float(np.std(x, ddof=1)), per_seed=[float(v) for v in x])

def distribution(x):
    x = np.asarray(x)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                positive=int((x>1e-8).sum()), negative=int((x < -1e-8).sum()), ties=int((abs(x)<=1e-8).sum()))

def main():
    started=time.monotonic()
    plan=read(OUT/'protocol.json')
    assert plan == make_plan(), 'Locked inputs/code/checkpoints changed'
    queue=read(CONTROL/'queue.json')
    assert queue['state']=='completed' and queue['error'] is None
    expected={f'{r}/{c}' for r in plan['runs'] for c in CONDITIONS}
    assert set(queue['jobs'])==expected and all(j['status']=='completed' for j in queue['jobs'].values())
    assert queue['plan_digest']==plan['digest']
    formal=read(FORMAL/'experiment_plan.json')
    machine=read(OUT/'diagnostic_results.json')
    assert machine['complete'] and machine['completed_jobs']==54 and machine['plan_digest']==plan['digest']
    summaries, all_rows, ref_maxdiff={}, {}, 0.
    n_artifacts=0
    replay_count=0
    no_change_count=0
    subset_counts={}
    image_hashes=set()
    for run in plan['runs']:
        cfg=formal['signatures'][run]['config']
        val=Path(cfg['DATASET']['VAL_PATH'])
        prompt_records=read_text(Path(cfg['DATASET']['TEXT_PROMPT_PATH'])/'Val_text.xlsx')
        prompt_map={r['Ground Truth']:r for r in prompt_records}
        refdir=FORMAL/'evaluations'/run/'Val/Kvasir'
        refs={r['name']:r for r in rows(refdir/'cases.csv')}
        bases={}
        for name,r in refs.items():
            with np.load(refdir/r['probability_file']) as z:
                bases[name]=z['mean'].copy()
        threshold=read(refdir/'summary.json')['source_train_area_threshold']
        for condition in CONDITIONS:
            key=f'{run}/{condition}'
            folder=OUT/'evaluations'/run/condition
            manifest=read(folder/'manifest.json')
            signature=dict(plan_digest=plan['digest'],run=run,condition=condition,checkpoint_sha256=plan['checkpoints'][run]['sha256'])
            assert manifest['complete'] and manifest['n']==100 and manifest['signature']==signature
            assert len(manifest['outputs'])==202
            for name,h in manifest['outputs'].items():
                assert sha256(folder/name)==h, (key,name)
                n_artifacts+=1
            saved=read(folder/'summary.json')
            assert saved==machine['summaries'][key]
            assert saved['mc_samples']==30 and saved['batch_size']==32 and saved['split']=='Val' and saved['diagnostic']
            assert saved['checkpoint_sha256']==plan['checkpoints'][run]['sha256']
            raw=rows(folder/'cases.csv')
            assert len(raw)==100 and {r['name'] for r in raw}==set(refs)
            actual=[]
            for r in raw:
                name=r['name']
                assert r['image_name']==refs[name]['image_name']==prompt_map[name]['Image']
                assert r['condition']==condition
                record=prompt_map[name]
                # Formal loader uses the exact prompt stored in the source sheet.
                text_field=next(k for k in record if k not in ('Image','Ground Truth'))
                assert r['original_prompt']==record[text_field], (text_field,record)
                assert r['input_prompt']==prompt_for(r['original_prompt'],condition)
                changed=r['input_prompt']!=r['original_prompt']
                direction=direction_bearing(r['original_prompt'])
                assert (r['prompt_changed']=='True')==changed
                assert (r['direction_bearing']=='True')==direction
                with np.load(folder/r['probability_file']) as z:
                    assert set(z.files)=={'mean','variance','entropy'}
                    p,v,e=(z[k] for k in ('mean','variance','entropy'))
                    assert all(x.shape==(224,224) and x.dtype==np.float32 and np.isfinite(x).all() for x in (p,v,e))
                    assert p.min()>=0 and p.max()<=1 and v.min()>=0 and v.max()<=.259
                    entropy=-(p*np.log(p+1e-8)+(1-p)*np.log(1-p+1e-8))
                    assert np.allclose(e,entropy,atol=1e-6,rtol=1e-5)
                png=cv2.imread(str(folder/r['mask_file']),0)
                assert np.array_equal(png, (p>.5).astype(np.uint8)*255)
                base=bases[name]
                if condition=='original' or (condition=='opposite_location' and not changed):
                    assert np.allclose(p,base,atol=1e-5,rtol=1e-5) and np.array_equal(p>.5,base>.5)
                    ref_maxdiff=max(ref_maxdiff,float(abs(p-base).max()))
                    if condition=='original': replay_count+=1
                    else: no_change_count+=1
                metrics=case_metrics(val/'label'/name,p)
                for k,value in metrics.items(): close(value,r[k],(key,name,k))
                mae=float(abs(p-base).mean())
                disagreement=float(((p>.5)!=(base>.5)).mean())
                original=float(refs[name]['DSC'])
                for k,value in dict(aligned_probability_mae=mae,binary_disagreement_fraction=disagreement,
                                    original_DSC=original,delta_DSC_pp=100*(metrics['DSC']-original)).items():
                    close(value,r[k],(key,name,k))
                metrics.update(name=name, image_name=r['image_name'], prompt_changed=changed,direction_bearing=direction,
                               aligned_probability_mae=mae,binary_disagreement_fraction=disagreement,
                               delta_DSC_pp=100*(metrics['DSC']-original))
                actual.append(metrics)
            recalculated=summarize(actual,threshold)
            for k,value in recalculated.items(): close(value,saved[k],(key,k))
            for subset,chosen in [('all',actual),('direction_bearing',[r for r in actual if r['direction_bearing']]),
                                  ('prompt_changed',[r for r in actual if r['prompt_changed']])]:
                expected_sub=saved['subsets'][subset]
                assert expected_sub['n']==len(chosen)
                for source,dest in [('delta_DSC_pp','mean_delta_DSC_pp'),('aligned_probability_mae','aligned_probability_mae'),
                                    ('binary_disagreement_fraction','binary_disagreement_fraction')]:
                    close(float(np.mean([r[source] for r in chosen])) if chosen else None,expected_sub[dest],(key,subset,dest))
            assert saved['prompt_changed_count']==sum(r['prompt_changed'] for r in actual)
            subset_counts[key]={k:v['n'] for k,v in saved['subsets'].items()}
            summaries[key]=saved
            all_rows[key]={r['name']:r for r in actual}
            print('PASS',key,'100 predictions / full metrics / hashes',flush=True)
        if not image_hashes:
            image_hashes={sha256(val/'img'/r['Image']) for r in prompt_records}
    seeds=(666,667,668)
    contrasts={}
    table=[]
    for condition in CONDITIONS:
        item={}
        for metric in ('DSC_percent','small_DSC_percent','NSD_percent','micro_foreground_precision_percent','micro_foreground_recall_percent',
                       'empty_positive','zero_overlap_positive','positive_DSC_below_50_percent'):
            per_arm={arm:moments([summaries[f'{arm}_seed{s}/{condition}'][metric] for s in seeds])
                     for arm in ('paired_repeat','paired_image','paired_aug')}
            values=[summaries[f'paired_aug_seed{s}/{condition}'][metric]-summaries[f'paired_image_seed{s}/{condition}'][metric] for s in seeds]
            basevalues=[summaries[f'paired_aug_seed{s}/original'][metric]-summaries[f'paired_image_seed{s}/original'][metric] for s in seeds]
            item[metric]=dict(arms=per_arm,aug_minus_image=moments(values),change_vs_original=moments(np.subtract(values,basevalues)))
            if metric in machine['aug_minus_image_pp'][condition]:
                close(np.mean(values),machine['aug_minus_image_pp'][condition][metric])
            if metric in machine['aggregates'][condition]['paired_aug']:
                for arm in per_arm:
                    close(per_arm[arm]['mean'],machine['aggregates'][condition][arm][metric]['mean'])
                    close(per_arm[arm]['sample_sd'],machine['aggregates'][condition][arm][metric]['sample_std'])
            table.append(dict(condition=condition,metric=metric,**{arm:per_arm[arm]['mean'] for arm in per_arm},
                              delta_mean=np.mean(values),delta_seed666=values[0],delta_seed667=values[1],delta_seed668=values[2],
                              change_in_advantage=np.mean(np.subtract(values,basevalues))))
        item['subsets']={}
        for subset in ('all','direction_bearing','prompt_changed','small'):
            names=[n for n,r in all_rows[f'paired_image_seed666/{condition}'].items()
                   if subset=='all' or (subset=='small' and r['gt_fraction']<=threshold)
                   or (subset in ('direction_bearing','prompt_changed') and r[subset])]
            if not names:
                item['subsets'][subset]={'n':0}
                continue
            entries={}
            for metric in ('DSC','aligned_probability_mae','binary_disagreement_fraction'):
                scale=100 if metric=='DSC' else 1
                d=np.array([[scale*(all_rows[f'paired_aug_seed{s}/{condition}'][n][metric]-all_rows[f'paired_image_seed{s}/{condition}'][n][metric])
                             for n in names] for s in seeds])
                entries[metric]=dict(seed_means=moments(d.mean(axis=1)),image_averages=distribution(d.mean(axis=0)))
            item['subsets'][subset]=dict(n=len(names),metrics=entries)
        contrasts[condition]=item
    with (HERE/'paired_contrasts.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    write_json('analysis.json',contrasts)
    audit=dict(passed=True,checked_at=datetime.now().astimezone().isoformat(),scope='CPU artifacts/metrics audit; no new inference or training',
               plan_digest=plan['digest'],jobs=54,predictions_recomputed=5400,output_hashes_verified=n_artifacts,
               formal_and_diagnostic_locked_inputs_reverified=True,original_replays=replay_count,unchanged_opposite_replays=no_change_count,
               maximum_replay_probability_difference=ref_maxdiff,unique_val_image_hashes=len(image_hashes),
               queue_completed_at=queue['completed_at'],first_job_started_at=min(j['started_at'] for j in queue['jobs'].values()),
               active_seconds=sum(j['active_seconds'] for j in queue['jobs'].values()),cpu_audit_seconds=time.monotonic()-started,
               n_negative=0,n_small=25,subset_counts=subset_counts,
               result_sources={p:sha256(OUT/p) for p in ('protocol.json','diagnostic_results.json','diagnostic_report.md')},
               limitations=['Source Val reused for development; not independent confirmation',
                            'Fixed weights saw attribute prompts during training; generic intervention is not retrained class-only baseline',
                            'MC variance accepted as saved finite arrays; cannot regenerate draw variance without inference',
                            'No inference of patient independence or clinical applicability'])
    write_json('audit.json',audit)
    print('AUDIT PASSED', json.dumps({k:v for k,v in audit.items() if k not in ('subset_counts','result_sources','limitations')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
