"""CPU-only verification of completed seed667 evaluations; preserves the paused queue."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json,sys,csv,hashlib,gc
from datetime import datetime,timezone,timedelta
import numpy as np
import cv2
cv2.setNumThreads(1)
HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
ROOT=WORKSPACE/'MedCLIPSeg'
sys.path.insert(0,str(ROOT))
from research_formal.metrics import case_metrics,summarize
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
out=ROOT/'outputs_research/formal_kvasir_v1'
plan=read(out/'experiment_plan.json')
queue=read(WORKSPACE/'control/research_formal/formal_kvasir_v1/queue.json')
summaries,cases,checks={},{},{}
for variant in ('paired_repeat','paired_image','paired_aug'):
 run=variant+'_seed667'; folder=out/'evaluations'/run/'Val/Kvasir'
 assert queue['jobs']['train/'+run]['status']==queue['jobs']['val/'+run]['status']=='completed'
 s,m=read(folder/'summary.json'),read(folder/'manifest.json')
 assert m['complete'] and s['n']==100 and s['n_small']==25 and s['mc_samples']==30 and s['split']=='Val'
 assert m['signature']['training_signature']==plan['signatures'][run]['digest']
 assert sha(folder/'cases.csv')==m['cases_sha256']
 rows=list(csv.DictReader((folder/'cases.csv').open(encoding='utf-8-sig',newline='')))
 recalculated=[]
 for row in rows:
  with np.load(folder/row['probability_file']) as z:
   for key in ('mean','variance','entropy'):
    assert z[key].shape==(224,224) and np.isfinite(z[key]).all()
   mean=z['mean'].copy()
   assert mean.min()>=0 and mean.max()<=1 and z['variance'].min()>=0
  mask=cv2.imread(str(folder/row['mask_file']),0)
  assert np.array_equal(mask,(mean>.5).astype(np.uint8)*255)
  metrics=case_metrics(ROOT/'data/Kvasir/Val_Folder/label'/row['name'],mean)
  for key,value in metrics.items():
   if value is None:assert row[key]==''
   else:assert np.isclose(float(row[key]),value,atol=1e-8,rtol=1e-6),(key,row['name'])
  recalculated.append(dict(name=row['name'],**metrics))
 recomputed=summarize(recalculated,s['source_train_area_threshold'])
 for key,value in recomputed.items():
  if value is None:assert s[key] is None
  else:assert np.allclose(value,s[key],atol=1e-8,rtol=1e-6),key
 checks[variant]=dict(new_saved_predictions_recomputed=100)
 summaries[variant]=s
 cases[variant]={r['name']:{k:(float(v) if k in ('DSC','foreground_recall','foreground_precision','gt_fraction') and v else v) for k,v in r.items()} for r in rows}
 print('VERIFIED saved evaluation',variant,flush=True)

# Load completed seed667 checkpoints only, on CPU.
import torch
torch.set_num_threads(1)
initial=[]
# Check current code and source data against the locked signatures.
reference=plan['signatures']['paired_repeat_seed667']
for name,h in reference['code'].items():
 assert sha(ROOT/name)==h,name
for name,h in reference['data'].items():
 assert sha(Path(name))==h,name
print('VERIFIED current code and Train/Val data hashes',flush=True)
for variant in summaries:
 run=variant+'_seed667'; p=out/run/'Kvasir/trained_models/seed667/MedCLIPSeg_unimedclip_ViT-B-16_latest.pth'
 assert sha(p)==summaries[variant]['checkpoint_sha256']
 checkpoint=torch.load(p,map_location='cpu',weights_only=False)
 assert checkpoint['epoch']==99 and checkpoint['progress']['phase']=='epoch_complete'
 assert checkpoint['scheduler']['last_epoch']==100
 steps={int(s['step']) for s in checkpoint['optimizer']['state'].values()}
 assert steps=={3400},steps
 assert checkpoint['research_signature']['digest']==plan['signatures'][run]['digest']
 assert all(not v.is_floating_point() or torch.isfinite(v).all().item() for v in checkpoint['model'].values())
 assert all(not isinstance(v,torch.Tensor) or not v.is_floating_point() or torch.isfinite(v).all().item() for state in checkpoint['optimizer']['state'].values() for v in state.values())
 initial.append(checkpoint['initial_fingerprints'])
 checks[variant].update(epoch=100,adam_steps=3400,scheduler_steps=100,checkpoint_sha256=summaries[variant]['checkpoint_sha256'])
 del checkpoint
 gc.collect()
 print('VERIFIED final checkpoint',variant,flush=True)
assert initial[0]==initial[1]==initial[2]

keys=['DSC_percent','small_DSC_percent','NSD_percent','micro_foreground_recall_percent','micro_foreground_precision_percent','positive_DSC_below_50_percent','zero_overlap_positive']
comparisons={}
for a,b in [('paired_image','paired_repeat'),('paired_aug','paired_repeat'),('paired_aug','paired_image')]:
 rows=[dict(name=n,small=cases[a][n]['gt_fraction']<=summaries[a]['source_train_area_threshold'],delta_pp=100*(cases[a][n]['DSC']-cases[b][n]['DSC'])) for n in cases[a]]
 def distribution(rs):
  values=np.array([r['delta_pp'] for r in rs])
  return dict(n=len(rs),improved=int((values>1e-8).sum()),declined=int((values< -1e-8).sum()),tied=int((abs(values)<=1e-8).sum()),median_pp=float(np.median(values)))
 ordered=sorted(rows,key=lambda r:r['delta_pp'],reverse=True)
 comparisons[a+'_vs_'+b]=dict(delta={k:summaries[a][k]-summaries[b][k] for k in keys},
  all_cases=distribution(rows),small_cases=distribution([r for r in rows if r['small']]),
  top_two_contribution_pp=sum(r['delta_pp'] for r in ordered[:2])/100,
  remaining_98_mean_pp=float(np.mean([r['delta_pp'] for r in ordered[2:]])),
  best=ordered[:3],worst=ordered[-3:])
hard=['cju40w3hbkwpn08015rbs3wko.png','cju76o55nymqd0871h31sph9w.png','cju89z6pqpqfx0817mfv8ixjc.png']
result=dict(checked_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),passed=True,
 scope='seed667 source Val only; 300 predictions recomputed; three final checkpoints and code/data hashes verified; no GPU inference',
 summaries=summaries,checks=checks,same_initial_fingerprints=True,comparisons=comparisons,
 hard_cases={n:{v:cases[v][n]['DSC']*100 for v in summaries} for n in hard},
 queue_snapshot=queue,plan_digest=plan['digest'],initial_fingerprints=initial[0],code_and_data_hashes_verified=True,model_and_optimizer_finite=True)
previous=read(WORKSPACE/'reports/formal_seed666_20261002/audit.json')
for variant, prior in previous['summaries'].items():
 assert read(out/'evaluations'/(variant+'_seed666')/'Val/Kvasir/summary.json')==prior
 folder=out/'evaluations'/(variant+'_seed666')/'Val/Kvasir'
 manifest=read(folder/'manifest.json')
 assert sha(folder/'cases.csv')==manifest['cases_sha256']
result['two_seed_aggregates']={v:{k:float(np.mean([previous['summaries'][v][k],summaries[v][k]])) for k in keys} for v in summaries}
result['two_seed_comparisons']={name:{k:float(np.mean([previous['comparisons'][name]['delta'][k],value['delta'][k]])) for k in keys} for name,value in comparisons.items()}
result['previous_seed_audit_sha256']=sha(WORKSPACE/'reports/formal_seed666_20261002/audit.json')
control=WORKSPACE/'control/research_formal/formal_kvasir_v1'
paused=read(control/'trainer_status.json')
assert read(control/'queue.json')['state']=='paused' and paused['status']=='paused' and (control/'PAUSE.request').exists()
p=Path(paused['checkpoint'])
checkpoint=torch.load(p,map_location='cpu',weights_only=False)
assert checkpoint['epoch']==paused['epoch']-1 and checkpoint['progress']['phase']=='train'
assert checkpoint['progress']['next_batch_index']==paused['step']
assert checkpoint['research_signature']['digest']==plan['signatures']['paired_aug_seed668']['digest']
assert all(k in checkpoint for k in ('optimizer','scheduler','rng_state','progress','runtime'))
steps={int(v['step']) for v in checkpoint['optimizer']['state'].values()}
assert steps=={41} and checkpoint['scheduler']['last_epoch']==1
result['pause_checkpoint']=dict(epoch=paused['epoch'],completed_batches_in_epoch=paused['step'],adam_steps=41,scheduler_steps=1,checkpoint_sha256=sha(p),rng_and_runtime_present=True,queue_state='paused')
del checkpoint
gc.collect()
print('VERIFIED paused seed668 checkpoint: epoch 2, batch 7, Adam step 41',flush=True)
(HERE/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(comparisons=comparisons,hard_cases=result['hard_cases']),ensure_ascii=False,indent=2))
