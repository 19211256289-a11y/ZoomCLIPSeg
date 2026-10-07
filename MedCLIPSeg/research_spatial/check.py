"""Exact neutral baseline, real B24 Adam steps, strict restore, CPU pause tests."""
import gc,hashlib,subprocess,sys,time
import torch,monai
from torch.utils.data import DataLoader
from train_spatial import set_random_seed,forward_batch,calc_loss,atomic_save_checkpoint
from .model import build_model,parameter_fingerprints
from utils.main_utils import read_text
from .common import ROOT,MODES,make_config,control_dir,suite_output,now,write_json,object_hash,code_manifest,optimizer_groups,read,sha256
from .data import make_dataset
from .objective import paired_objective


def base_fingerprints(model):
    digests={k:hashlib.sha256() for k in ('frozen','trainable')}
    for name,p in model.named_parameters():
        if name.startswith('spatial_query.'):continue
        value=p.detach().cpu().contiguous();h=digests['trainable' if p.requires_grad else 'frozen']
        h.update(f'{name}|{value.dtype}|{list(value.shape)}'.encode());h.update(value.numpy().tobytes())
    return {k:v.hexdigest() for k,v in digests.items()}


def run_checks(suite):
    destination=control_dir(suite)/'check.json';write_json(destination,dict(passed=False,started_at=now()))
    directory=suite_output(suite)/'checks';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'cpu.log').open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-u','-m','research_spatial.tests'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('CPU tests failed; inspect checks/cpu.log')
    reports={};initial={};reference=None;neutral_mc=None;neutral_loss=None
    old_initial=read(ROOT.parent/'control/research_prompt/pilot_prompt_kvasir_v1/check.json')['initial_fingerprints']
    for mode in MODES:
        cfg=make_config(suite,mode+'_seed666');set_random_seed(666);model=build_model(cfg)
        initial[mode]=parameter_fingerprints(model);native=base_fingerprints(model)
        if reference is None:reference=native
        assert native==reference==old_initial,'Native initialization differs from historical baseline'
        model.to(cfg.MODEL.DEVICE)
        data=make_dataset(cfg,'Train',read_text(ROOT/'data/Kvasir/Prompts_Folder/Train_text.xlsx')[:24])
        set_random_seed(666);batch=next(iter(DataLoader(data,batch_size=24,shuffle=False,num_workers=0)))
        short={k:v[:2] for k,v in batch.items()}
        model.eval();set_random_seed(666)
        with torch.no_grad():mc=forward_batch(model,short,cfg.MODEL.DEVICE,2).cpu()
        if neutral_mc is None:neutral_mc=mc
        assert torch.equal(neutral_mc,mc),'Neutral initialization must exactly preserve native MC logits'
        model.train();set_random_seed(666)
        optimizer=torch.optim.Adam(optimizer_groups(model,cfg),lr=cfg.TRAIN.LEARNING_RATE)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=suite['epochs'],eta_min=cfg.TRAIN.MIN_LEARNING_RATE)
        ce=torch.nn.BCEWithLogitsLoss();dice=monai.losses.DiceLoss(include_background=True,sigmoid=True)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        losses=[]
        for step in range(2):
            optimizer.zero_grad();loss,diag=paired_objective(model,batch,cfg,ce,dice,forward_batch,calc_loss)
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if mode!='baseline':
                assert model.spatial_query.experts[-1].weight.grad.abs().sum()>0
                if step:
                    assert model.spatial_query.experts[1].weight.grad.abs().sum()>0
                    assert model.spatial_query.router[-1].weight.grad.abs().sum()>0
            losses.append(float(loss.detach()))
            if step==0:
                if neutral_loss is None:neutral_loss=losses[0]
                assert losses[0]==neutral_loss,'Initial paired loss differs'
            optimizer.step()
        scheduler.step();torch.cuda.synchronize();seconds=time.perf_counter()-start
        after=parameter_fingerprints(model)
        assert after['frozen']==initial[mode]['frozen'] and after['trainable']!=initial[mode]['trainable']
        path=directory/f'{mode}_disposable.pth'
        atomic_save_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),str(path))
        loaded=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(loaded['model'],strict=True);optimizer.load_state_dict(loaded['optimizer']);scheduler.load_state_dict(loaded['scheduler'])
        assert parameter_fingerprints(model)==after
        reports[mode]=dict(losses=losses,seconds_per_batch=seconds/2,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,checkpoint_roundtrip=True,frozen_unchanged=True,
                           neutral_mc_exact=True,neutral_loss_exact=True,module_gradients_second_step=True,
                           parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('PASS GPU B24 and neutral equivalence',mode,reports[mode],flush=True)
        del loaded,model,optimizer,scheduler,loss,batch,short,mc,data;gc.collect();torch.cuda.empty_cache();path.unlink()
    assert initial['global_mix']==initial['spatial_mix']
    assert reports['global_mix']['parameters']==reports['spatial_mix']['parameters']
    for previous in ('formal_kvasir_v1','pilot_prompt_kvasir_v1','pilot_softcon_kvasir_v1','pilot_query_kvasir_v1'):
        plan=read(ROOT/'outputs_research'/previous/'experiment_plan.json')
        for name,h in next(iter(plan['signatures'].values()))['code'].items():assert sha256(ROOT/name)==h
    report=dict(passed=True,completed_at=now(),suite=suite['name'],suite_sha256=sha256(ROOT/'research_spatial/suites'/f'{suite["name"]}.yaml'),
                code_digest=object_hash(code_manifest()),cpu_tests=7,cpu_production_pause_resume_all_modes=True,
                gpu_checks=reports,native_initialization_matches_prior_baseline=True,initial_fingerprints=initial,
                previous_signed_code_unchanged=True,scientific_training_started=False)
    write_json(destination,report);print('ALL CHECKS PASSED',flush=True);return report
