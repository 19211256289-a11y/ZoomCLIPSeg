"""CPU production pause regression and disposable GPU checks for all Formal arms."""
import gc, json, subprocess, sys, time
import monai
import torch
from torch.utils.data import DataLoader
from train_formal import set_random_seed, forward_batch, calc_loss, capture_rng_state, atomic_save_checkpoint
from .objective import paired_objective
from .common import (ROOT, make_config, control_dir, suite_output, write_json, object_hash,
                     code_manifest, optimizer_groups, sha256, now, VARIANTS)
from .model import build_model, parameter_fingerprints
from .data import make_dataset
from utils.main_utils import read_text


def gpu_probe(suite):
    cfg = make_config(suite, 'paired_aug_seed666', True)
    data = make_dataset(cfg, 'Train', read_text(ROOT/'data/Kvasir/Prompts_Folder/Train_text.xlsx')[:72])
    set_random_seed(666)
    model = build_model(cfg)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_fingerprints = parameter_fingerprints(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    reports = {}
    for variant in VARIANTS:
        cfg = make_config(suite, variant+'_seed666', True)
        set_random_seed(666)
        model = build_model(cfg)
        assert parameter_fingerprints(model) == initial_fingerprints, 'Same-seed initialization differs between arms'
        model.train().cuda()
        set_random_seed(666)
        optimizer = torch.optim.Adam(optimizer_groups(model, cfg), lr=cfg.TRAIN.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=cfg.TRAIN.MIN_LEARNING_RATE)
        ce, dice = torch.nn.BCEWithLogitsLoss(), monai.losses.DiceLoss(include_background=True, sigmoid=True)
        times, diagnostics = [], []
        torch.cuda.reset_peak_memory_stats()
        for batch in DataLoader(data, batch_size=24, shuffle=False, num_workers=0):
            optimizer.zero_grad()
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss, diag = paired_objective(model, batch, cfg, ce, dice, forward_batch, calc_loss)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
            diagnostics.append(diag)
            print('GPU check', variant, len(times), '/3', round(times[-1], 3), flush=True)
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                assert torch.equal(parameter.detach().cpu(), initial[name]), name
        changed = sum(not torch.equal(p.detach().cpu(), initial[n]) for n, p in model.named_parameters() if p.requires_grad)
        assert changed > 0
        path = suite_output(suite)/'checks/disposable_probe.pth'
        atomic_save_checkpoint(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                    scheduler=scheduler.state_dict(), rng_state=capture_rng_state()), str(path))
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert all(torch.equal(p.cpu(), saved['model'][n]) for n, p in model.state_dict().items())
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        assert all(int(s['step']) == 3 for s in optimizer.state.values())
        model.eval()
        small = {k: v[:2] if isinstance(v, (list, torch.Tensor)) else v for k, v in batch.items()}
        with torch.inference_mode():
            output = forward_batch(model, small, 'cuda', num_samples=2)
        assert output.shape == (2, 2, 224, 224) and torch.isfinite(output).all()
        reports[variant] = dict(steps=3, batch_size=24, step_seconds=times, diagnostics=diagnostics,
                                initial_fingerprints=initial_fingerprints, fresh_from_base=True,
                                frozen_weights_unchanged=True, updated_trainable_tensors=changed,
                                checkpoint_round_trip=True, single_view_inference=True,
                                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        del saved, optimizer, scheduler, loss, output, model, parameter
        gc.collect()
        torch.cuda.empty_cache()
        path.unlink()
    seeds = {666: initial_fingerprints}
    for seed in (667, 668):
        cfg = make_config(suite, f'paired_aug_seed{seed}', True)
        set_random_seed(seed)
        model = build_model(cfg)
        seeds[seed] = parameter_fingerprints(model)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    assert len({v['trainable'] for v in seeds.values()}) == 3, 'Fresh seeds must change adapters/head'
    assert len({v['frozen'] for v in seeds.values()}) == 1, 'Foundation encoders must be identical'
    # Strict loader must reproduce the original native initialization exactly.
    from trainers.medclipseg_unimedclip import CustomCLIP, load_unimedclip_to_device
    cfg = make_config(suite, 'paired_aug_seed666', True)
    set_random_seed(666)
    original = CustomCLIP(cfg, load_unimedclip_to_device(cfg).float())
    for name, param in original.named_parameters():
        param.requires_grad_(name.startswith(('pvl_adapters.', 'mask_head.', 'upscale.')))
    assert parameter_fingerprints(original) == initial_fingerprints, 'Formal initialization differs from native foundation initialization'
    reports['initialization'] = dict(per_seed=seeds, same_seed_arms_identical=True,
                                     different_seed_heads_distinct=True, native_loader_equivalent=True)
    del original, initial, param
    gc.collect()
    torch.cuda.empty_cache()
    return reports


def run_checks(suite):
    destination = control_dir(suite)/'check.json'
    write_json(destination, dict(passed=False, state='running', started_at=now()))
    logs = suite_output(suite)/'checks'
    logs.mkdir(parents=True, exist_ok=True)
    try:
        with (logs/'cpu.log').open('w', encoding='utf-8') as stream:
            cpu = subprocess.run([sys.executable, '-u', '-m', 'research_formal.tests'], cwd=ROOT,
                                 stdout=stream, stderr=subprocess.STDOUT)
        if cpu.returncode:
            raise RuntimeError('CPU regression failed; see '+str(logs/'cpu.log'))
        print('PASS CPU matched views, gradients, RNG, seed paths and production pause/resume', flush=True)
        from .common import test_cohort_manifest
        cohorts = test_cohort_manifest(suite)
        print('PASS locked test file integrity and source Train/Val disjointness:', {k:v['n'] for k,v in cohorts.items()}, flush=True)
        arms = gpu_probe(suite)
        origin = json.loads((ROOT/'research_formal/ORIGIN.json').read_text())
        assert all(sha256(ROOT/p) == h for p, h in origin.items()), 'Earlier source changed'
        report = dict(passed=True, suite=suite['name'], batch_size=24, gradient_checkpointing=True,
                      cpu_regressions_passed=True, arm_checks=arms,
                      previous_research_code_unchanged=True, code_digest=object_hash(code_manifest()),
                      test_cohort_sizes={k:v['n'] for k,v in cohorts.items()},
                      completed_at=now(), training_started=False)
        write_json(destination, report)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        write_json(destination, dict(passed=False, error=str(exc), failed_at=now()))
        raise
