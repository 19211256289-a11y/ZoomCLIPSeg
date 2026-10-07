"""Short real-batch checks; never writes a scientific checkpoint."""
import gc,subprocess,sys,time
import torch
import monai
from torch.utils.data import DataLoader
from train_prompt import set_random_seed,forward_batch,calc_loss,atomic_save_checkpoint
from research_formal.model import build_model,parameter_fingerprints
from utils.main_utils import read_text
from .common import ROOT,MODES,make_config,control_dir,suite_output,now,write_json,object_hash,code_manifest,optimizer_groups,read,sha256
from .data import make_dataset
from .objective import paired_objective

def run_checks(suite):
    destination=control_dir(suite)/'check.json'
    write_json(destination,dict(passed=False,started_at=now()))
    directory=suite_output(suite)/'checks';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'cpu.log').open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-u','-m','research_prompt.tests'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('CPU tests failed; inspect checks/cpu.log')
    reports={};initial=None
    for mode in MODES:
        cfg=make_config(suite,mode+'_seed666')
        set_random_seed(666);model=build_model(cfg)
        fingerprints=parameter_fingerprints(model)
        if initial is None:initial=fingerprints
        assert initial==fingerprints,'Initialization differs across modes'
        model.train().to(cfg.MODEL.DEVICE)
        data=make_dataset(cfg,'Train',read_text(ROOT/'data/Kvasir/Prompts_Folder/Train_text.xlsx')[:24])
        set_random_seed(666)
        batch=next(iter(DataLoader(data,batch_size=24,shuffle=False,num_workers=0)))
        optimizer=torch.optim.Adam(optimizer_groups(model,cfg),lr=cfg.TRAIN.LEARNING_RATE)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=suite['epochs'],eta_min=cfg.TRAIN.MIN_LEARNING_RATE)
        ce=torch.nn.BCEWithLogitsLoss();dice=monai.losses.DiceLoss(include_background=True,sigmoid=True)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        optimizer.zero_grad()
        loss,diag=paired_objective(model,batch,cfg,ce,dice,forward_batch,calc_loss)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
        optimizer.step();scheduler.step();torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        after=parameter_fingerprints(model)
        assert after['frozen']==initial['frozen'] and after['trainable']!=initial['trainable']
        path=directory/f'{mode}_disposable.pth'
        atomic_save_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),str(path))
        loaded=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(loaded['model'],strict=True);optimizer.load_state_dict(loaded['optimizer']);scheduler.load_state_dict(loaded['scheduler'])
        assert parameter_fingerprints(model)==after
        reports[mode]=dict(loss=float(loss.detach()),seconds_per_batch=seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,diagnostics=diag,checkpoint_roundtrip=True,
                           frozen_unchanged=True,trainable_updated=True)
        print('PASS GPU batch24 forward/backward/Adam/save/load',mode,reports[mode],flush=True)
        del loaded,model,optimizer,scheduler,loss,batch,data
        gc.collect();torch.cuda.empty_cache();path.unlink()
    # Ensure old scientific code still matches the completed protocol.
    prior=read(ROOT/'outputs_research/formal_kvasir_v1/experiment_plan.json')
    reference=next(iter(prior['signatures'].values()))
    assert all(sha256(ROOT/name)==h for name,h in reference['code'].items())
    report=dict(passed=True,completed_at=now(),suite=suite['name'],suite_sha256=sha256(ROOT/'research_prompt/suites'/f'{suite["name"]}.yaml'),
                code_digest=object_hash(code_manifest()),cpu_tests=4,cpu_production_pause_resume_all_modes=True,
                gpu_checks=reports,same_seed_initialization_identical=True,initial_fingerprints=initial,
                previous_formal_code_unchanged=True,scientific_training_started=False)
    write_json(destination,report)
    print('ALL CHECKS PASSED; disposable optimizer steps are not scientific training.',flush=True)
    return report
