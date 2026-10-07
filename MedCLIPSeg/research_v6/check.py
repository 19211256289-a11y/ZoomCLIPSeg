"""Paired batch24 hardware probe and exact CPU pause recovery before training."""
import gc,json,subprocess,sys,time
import torch,monai
from torch.utils.data import DataLoader
from train_v6 import set_random_seed,forward_batch,calc_loss,capture_rng_state,atomic_save_checkpoint
from .objective import paired_objective
from .common import ROOT,make_config,control_dir,suite_output,write_json,object_hash,code_manifest,optimizer_groups,sha256,now
from .model import build_model,initialize_from_checkpoint
from .data import make_dataset
from utils.main_utils import read_text


def gpu_probe(suite,checkpointing):
    cfg=make_config(suite,'paired_consistency',checkpointing)
    data=make_dataset(cfg,'Train',read_text(ROOT/'data/Kvasir/Prompts_Folder/Train_text.xlsx')[:120])
    loader=DataLoader(data,batch_size=24,shuffle=False,num_workers=0)
    set_random_seed(666);model=build_model(cfg);initialize_from_checkpoint(model,cfg.RESEARCH.INIT_FROM);model.train().cuda()
    initial=torch.load(cfg.RESEARCH.INIT_FROM,map_location='cpu',weights_only=False)['model']
    assert set(initial)==set(model.state_dict())
    optimizer=torch.optim.Adam(optimizer_groups(model,cfg),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20,eta_min=1e-5)
    ce=torch.nn.BCEWithLogitsLoss();dice=monai.losses.DiceLoss(include_background=True,sigmoid=True)
    times=[];diagnostics=[];torch.cuda.reset_peak_memory_stats()
    for index,batch in enumerate(loader):
        optimizer.zero_grad();torch.cuda.synchronize();start=time.perf_counter()
        loss,d=paired_objective(model,batch,cfg,ce,dice,forward_batch,calc_loss)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step();torch.cuda.synchronize();times.append(time.perf_counter()-start);diagnostics.append(d)
        print('GPU paired check',index+1,'/5',round(times[-1],3),d,flush=True)
    for n,p in model.named_parameters():
        if not p.requires_grad:assert torch.equal(p.detach().cpu(),initial[n]),n
    probe=suite_output(suite)/'checks/disposable_probe.pth';probe.parent.mkdir(parents=True,exist_ok=True)
    atomic_save_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng_state=capture_rng_state()),str(probe))
    saved=torch.load(probe,map_location='cpu',weights_only=False)
    for n,p in model.state_dict().items():assert torch.equal(p.cpu(),saved['model'][n]),n
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    assert all(int(s['step'])==5 for s in optimizer.state.values())
    model.eval();small={k:v[:2] if isinstance(v,(list,torch.Tensor)) else v for k,v in batch.items()}
    with torch.inference_mode():out=forward_batch(model,small,'cuda',num_samples=2)
    assert out.shape==(2,2,224,224) and torch.isfinite(out).all()
    result=dict(passed=True,batch_size=24,gradient_checkpointing=checkpointing,steps=5,
        frozen_weights_unchanged=True,checkpoint_round_trip=True,single_view_inference_passed=True,
        step_seconds=times,step_diagnostics=diagnostics,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    del saved,initial,model,optimizer,scheduler;gc.collect();torch.cuda.empty_cache();probe.unlink()
    return result


def run_checks(suite):
    destination=control_dir(suite)/'check.json';write_json(destination,dict(passed=False,state='running',started_at=now()))
    logs=suite_output(suite)/'checks';logs.mkdir(parents=True,exist_ok=True)
    try:
        with (logs/'cpu.log').open('w',encoding='utf-8') as f:
            process=subprocess.run([sys.executable,'-u','-m','research_v6.tests'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if process.returncode:raise RuntimeError('CPU checks failed; see '+str(logs/'cpu.log'))
        print('PASS CPU transforms, gated gradients and exact pause/resume',flush=True)
        # Two live encoder graphs reserved >16 GiB in the first verified probe.
        # Fix checkpointing for BOTH arms to avoid Windows shared-memory spill.
        report=gpu_probe(suite,True)
        origin=json.loads((ROOT/'research_v6/ORIGIN.json').read_text())
        assert all(sha256(ROOT/p)==h for p,h in origin.items()),'Older source changed'
        report.update(suite=suite['name'],cpu_regressions_passed=True,previous_research_code_unchanged=True,
            code_digest=object_hash(code_manifest()),completed_at=now(),formal_training_started=False)
        write_json(destination,report);print(json.dumps(report,indent=2),flush=True)
    except BaseException as exc:
        write_json(destination,dict(passed=False,error=str(exc),failed_at=now()));raise
