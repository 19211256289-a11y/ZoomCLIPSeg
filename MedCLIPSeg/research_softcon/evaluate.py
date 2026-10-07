"""Final-epoch source-Val evaluation under two prespecified prompt conditions."""
import gc,time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_softcon import TrainingControl, TrainingPaused, set_random_seed, forward_batch
from research_formal.model import build_model
from research_formal.metrics import foreground_threshold,case_metrics,summarize,write_cases
from utils.main_utils import read_text
from .common import read,sha256,object_hash,write_json,now,make_signature,assert_signature
from .data import make_dataset

def cached(destination,signature):
    p=destination/'manifest.json'
    if not p.exists():return None
    manifest=read(p)
    if manifest['signature']!=signature:raise ValueError('Evaluation signature changed')
    if not manifest.get('complete'):return None
    if len(manifest['outputs'])!=202:raise ValueError('Incomplete cached artifact list')
    for name,h in manifest['outputs'].items():
        if sha256(destination/name)!=h:raise ValueError('Cached evaluation modified: '+name)
    return read(destination/'summary.json')

def evaluate(cfg,checkpoint,destination,control_directory,condition):
    if condition not in ('generic','original'):raise ValueError(condition)
    signature=make_signature(cfg)
    es=dict(training_signature=signature['digest'],checkpoint_sha256=sha256(checkpoint),condition=condition,
            mc_samples=30,batch_size=32,split='Val',target='Kvasir',threshold=.5)
    destination=Path(destination)
    done=cached(destination,es)
    if done:return done
    control=TrainingControl(control_directory,run_id=cfg.RESEARCH.RUN_ID,condition=condition,phase='evaluation')
    if control.pause_requested():raise TrainingPaused()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert_signature(saved['research_signature'],signature)
    if saved['epoch']!=cfg.TRAIN.NUM_EPOCHS-1 or saved['progress']['phase']!='epoch_complete':raise ValueError('Final checkpoint required')
    set_random_seed(cfg.seed)
    model=build_model(cfg)
    model.load_state_dict(saved['model'],strict=True)
    del saved
    model.eval().to(cfg.MODEL.DEVICE)
    data=make_dataset(cfg,'Val',read_text(Path(cfg.DATASET.TEXT_PROMPT_PATH)/'Val_text.xlsx'),condition)
    loader=DataLoader(data,batch_size=32,shuffle=False,num_workers=0)
    threshold=foreground_threshold(cfg)
    for sub in ('masks','probabilities'):(destination/sub).mkdir(parents=True,exist_ok=True)
    manifest=dict(signature=es,complete=False,started_at=now())
    write_json(destination/'manifest.json',manifest)
    set_random_seed(cfg.seed)
    torch.backends.cudnn.benchmark=True
    torch.cuda.reset_peak_memory_stats()
    rows,artifacts,elapsed=[],{},0.
    with torch.no_grad():
        for index,batch in enumerate(loader):
            if control.pause_requested():raise TrainingPaused()
            torch.cuda.synchronize();start=time.perf_counter()
            draws=torch.sigmoid(forward_batch(model,batch,cfg.MODEL.DEVICE,30))
            p=draws.mean(0);v=draws.var(0,unbiased=False)
            e=-(p*torch.log(p+1e-8)+(1-p)*torch.log(1-p+1e-8))
            torch.cuda.synchronize();elapsed+=time.perf_counter()-start
            p,v,e=(x.cpu().numpy() for x in (p,v,e))
            for b,name in enumerate(batch['mask_name']):
                assert np.isfinite(p[b]).all()
                png,npz=f'masks/{len(rows):06d}.png',f'probabilities/{len(rows):06d}.npz'
                if not cv2.imwrite(str(destination/png),(p[b]>.5).astype(np.uint8)*255):raise OSError('PNG save failed')
                np.savez_compressed(destination/npz,mean=p[b],variance=v[b],entropy=e[b])
                artifacts[png],artifacts[npz]=sha256(destination/png),sha256(destination/npz)
                rows.append(dict(name=name,image_name=batch['image_name'][b],mask_file=png,probability_file=npz,
                                 input_prompt=batch['text_prompt'][b],**case_metrics(Path(cfg.DATASET.VAL_PATH)/'label'/name,p[b])))
            control.update('running',step=index+1,total_steps=len(loader),cases_completed=len(rows))
            del draws
    if len(rows)!=100 or len({r['name'] for r in rows})!=100:raise ValueError('Incomplete Val')
    result=summarize(rows,threshold)
    result.update(run_id=cfg.RESEARCH.RUN_ID,prompt_mode=cfg.RESEARCH.PROMPT_MODE,seed=cfg.seed,condition=condition,
                  checkpoint_sha256=es['checkpoint_sha256'],split='Val',target='Kvasir',mc_samples=30,batch_size=32,
                  inference_seconds=elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,completed_at=now())
    write_cases(destination/'cases.csv',rows);write_json(destination/'summary.json',result)
    for name in ('cases.csv','summary.json'):artifacts[name]=sha256(destination/name)
    manifest.update(complete=True,n=100,outputs=artifacts,finished_at=now())
    write_json(destination/'manifest.json',manifest)
    control.update('completed',cases_completed=100)
    del model;gc.collect();torch.cuda.empty_cache()
    return result
