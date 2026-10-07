"""Bounded verification: CPU regressions and five disposable GPU updates."""
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import monai
import torch
from torch.utils.data import DataLoader

from train import calc_loss, set_random_seed, forward_batch, atomic_save_checkpoint, capture_rng_state
from utils.main_utils import read_text
from .common import (ROOT, WORKSPACE, now, make_config, control_dir, suite_output,
                     write_json, object_hash, code_manifest, optimizer_groups)
from .data import make_dataset
from .model import build_model, initialize_from_checkpoint, is_new_parameter


def gpu_probe(suite, checkpointing):
    cfg = make_config(suite, 'detail_beta', checkpointing)
    records = read_text(Path(cfg.DATASET.TEXT_PROMPT_PATH)/'Train_text.xlsx')[:120]
    dataset = make_dataset(cfg, 'Train', records)
    loader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    # Load the same real saved weights in both models and compare neutral forward.
    legacy_cfg = make_config(suite, 'legacy_continue', False)
    original = build_model(legacy_cfg)
    initialize_from_checkpoint(original, legacy_cfg.RESEARCH.INIT_FROM)
    small = {k: (v[:2] if isinstance(v, (torch.Tensor, list)) else v) for k,v in batch.items()}
    global_batch = {k:v for k,v in small.items() if k != 'image_detail'}
    original.eval().to('cuda')
    set_random_seed(771)
    with torch.no_grad(): reference = forward_batch(original, global_batch, 'cuda',2).cpu()
    del original
    gc.collect(); torch.cuda.empty_cache()
    model = build_model(cfg)
    missing = initialize_from_checkpoint(model, cfg.RESEARCH.INIT_FROM)
    model.to('cuda').eval()
    set_random_seed(771)
    with torch.no_grad(): actual = forward_batch(model, small, 'cuda',2).cpu()
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=1e-5)
    neutral_error = float((actual-reference).abs().max())
    del actual,reference
    model.train()
    set_random_seed(cfg.seed)
    optimizer = torch.optim.Adam(optimizer_groups(model,cfg),lr=cfg.TRAIN.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20,eta_min=1e-5)
    dice = monai.losses.DiceLoss(include_background=True,sigmoid=True)
    ce = torch.nn.BCEWithLogitsLoss()
    times, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for index,batch in enumerate(loader):
        if index >= 5: break
        torch.cuda.synchronize(); start=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits,softcon=forward_batch(model,batch,'cuda')
        loss=calc_loss(logits,batch['ground_truth_mask'].cuda(),ce,dice,cfg)+.1*softcon+.001*model.last_regularization
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite probe loss')
        loss.backward()
        frozen_with_grad=[n for n,p in model.named_parameters() if not p.requires_grad and p.grad is not None]
        if frozen_with_grad: raise AssertionError(f'Frozen encoder gradients: {frozen_with_grad[:3]}')
        if index >= 1:
            for prefix in ('detail_encoder.','detail_decoder.'):
                if not any(p.grad is not None and p.grad.abs().sum().item()>0 for n,p in model.named_parameters() if n.startswith(prefix)):
                    raise AssertionError(f'No gradient in {prefix}')
            if not any(p.grad is not None and p.grad.abs().sum().item()>0 for n,p in model.named_parameters() if '.beta_predictor.' in n):
                raise AssertionError('No beta predictor gradient')
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter()-start); losses.append(loss.item())
        print(f'GPU short check {index+1}/5: loss={losses[-1]:.5f}, seconds={times[-1]:.2f}',flush=True)
    probe_dir=suite_output(suite)/'checks'
    probe_dir.mkdir(parents=True,exist_ok=True)
    path=probe_dir/'disposable_probe.pth'
    atomic_save_checkpoint({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                            'scheduler':scheduler.state_dict(),'rng_state':capture_rng_state(),
                            'not_a_research_training_result':True},str(path))
    saved=torch.load(path,map_location='cpu',weights_only=False)
    for name,value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(),saved['model'][name]):
            raise AssertionError(f'Checkpoint round-trip mismatch: {name}')
    model.load_state_dict(saved['model'],strict=True)
    optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
    for state in optimizer.state.values():
        if int(state['step'].item()) != len(times): raise AssertionError('Adam step state mismatch')
    peak_allocated,peak_reserved=torch.cuda.max_memory_allocated()/2**30,torch.cuda.max_memory_reserved()/2**30
    result=dict(passed=True,gradient_checkpointing=checkpointing,steps=len(times),batch_size=24,
                neutral_max_abs_error=neutral_error,step_seconds=times,losses=losses,
                peak_allocated_gib=peak_allocated,peak_reserved_gib=peak_reserved,
                new_trainable_parameters=sum(p.numel() for n,p in model.named_parameters() if is_new_parameter(n)),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                initialization_missing_keys=missing,checkpoint_round_trip=True)
    del saved,model,optimizer,scheduler
    gc.collect(); torch.cuda.empty_cache()
    # This exact file is disposable validation output, never a usable experiment checkpoint.
    path.unlink()
    return result


def run_checks(suite):
    destination=control_dir(suite)/'check.json'
    write_json(destination,{'passed':False,'state':'running','started_at':now()})
    logs=suite_output(suite)/'checks'
    logs.mkdir(parents=True,exist_ok=True)
    try:
        for label,args in [('research_cpu',[sys.executable,'-u','-m','research.tests']),
                           ('legacy_pause',[sys.executable,'-u',str(WORKSPACE/'reports/test_training_pause.py')])]:
            with (logs/f'{label}.log').open('w',encoding='utf-8') as stream:
                result=subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(f'{label} failed; see {logs/(label+".log")}')
            print('PASS',label,flush=True)
        try:
            report=gpu_probe(suite,False)
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            print('Retrying short check with activation checkpointing for ALL variants.',flush=True)
            report=gpu_probe(suite,True)
        report.update(passed=True,completed_at=now(),code_digest=object_hash(code_manifest()),
                      suite=suite['name'],cpu_regressions_passed=True,formal_training_started=False)
        write_json(destination,report)
        write_json(logs/'verification.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='initialization_missing_keys'},indent=2),flush=True)
        print('Check complete. No formal training queue has been started.',flush=True)
        return report
    except BaseException as exc:
        write_json(destination,{'passed':False,'completed_at':now(),'error':f'{type(exc).__name__}: {exc}',
                                'formal_training_started':False})
        raise
