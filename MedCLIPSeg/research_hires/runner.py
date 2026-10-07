"""Queue for the native-resolution pilot: wait for the GPU, check, train eight runs, evaluate Val, report."""
from local_runtime import configure_local_models
configure_local_models()
import argparse
import gc
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

from .common import *

BATCHES = 34+5
MIN_FREE_GIB = 6


def process_alive(pid):
    try:
        if os.name == 'nt':
            listing = subprocess.run(['tasklist', '/FI', f'PID eq {int(pid)}', '/NH'], capture_output=True, text=True).stdout.lower()
            return str(int(pid)) in listing and 'python' in listing
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def lock_plan(suite):
    from research_detail_formal.common import code_manifest as accepted_code
    configs = {run: make_config(suite, run) for run in suite['runs']}
    science = dict(suite=suite, protocol_sha256=sha256(SUITE_FILE), configs={r: science_config(c) for r, c in configs.items()},
                   accepted_training_code=accepted_code(), pilot_code=code_manifest())
    plan = dict(science, digest=object_hash(science))
    path = OUTPUT/'experiment_plan.json'
    if path.exists() and read(path)['digest'] != plan['digest'] and any(checkpoint_path(suite, r).exists() for r in suite['runs']):
        raise ValueError('Protocol or pilot code changed after training began; use a new suite name')
    write_json(path, plan)
    for run, cfg in configs.items():
        config_path(run).parent.mkdir(parents=True, exist_ok=True)
        config_path(run).write_text(yaml.safe_dump(science_config(cfg), sort_keys=False), encoding='utf-8')
    return plan


def status():
    suite = load_suite()
    queue, progress = maybe(CONTROL/'queue.json'), maybe(CONTROL/'trainer_status.json')
    jobs = queue.get('jobs', {})
    done = lambda key: jobs.get(key, {}).get('status') == 'completed'
    state = queue.get('state', 'not_started')
    print('Suite:', NAME)
    print('Queue:', state, '| Pause requested:', (CONTROL/'PAUSE.request').exists())
    if state == 'waiting_for_gpu':
        print('Waiting for the GPU: another queue is still running. Training starts automatically when it finishes.')
    runs = suite['runs']
    print(f"Check: {'done' if done('check') else 'pending'} | trained {sum(done('train/'+r) for r in runs)}/{len(runs)}"
          f" | Val {sum(done('val/'+r) for r in runs)}/{len(runs)} | report {'done' if done('report') else 'pending'}")
    rates = [jobs[k]['active_seconds']/(suite['epochs']*BATCHES) for k in jobs if k.startswith('train/') and done(k) and jobs[k].get('uninterrupted')]
    rate = sum(rates)/len(rates) if rates else 1.7
    remaining = 0.
    current = queue.get('current_job')
    for run in runs:
        value = maybe(OUTPUT/'evaluations'/run/'Val/summary.json')
        if value:
            print(f"  {run}: Val DSC {value['DSC_percent']:.2f}, small {value['small_DSC_percent']:.2f}, NSD {value['NSD_percent']:.2f}")
        if not done('train/'+run):
            epochs = suite['epochs']
            if current == 'train/'+run and progress.get('run_id') == run and progress.get('epoch'):
                epochs -= max(0, progress['epoch']-1)
            remaining += epochs*BATCHES*rate
        remaining += 60*(not done('val/'+run))
    if current and state == 'running':
        line = f'Current: {current}'
        if progress.get('phase') in ('train', 'validation', 'epoch_complete') and progress.get('status') in ('running', 'starting'):
            line += f" | epoch {progress.get('epoch')}/{suite['epochs']}, {progress.get('phase')} batch {progress.get('step')}/{progress.get('total_steps')}"
        print(line)
    if state not in ('completed',):
        print(f'Estimated remaining once running: {remaining/3600:.1f} h')
    if queue.get('error'):
        print('ERROR:', queue['error'])
    if state == 'completed':
        print('Report:', REPORT/'report.md')
    print('Updated:', queue.get('updated_at', 'n/a'), '(UTC)')
    return state


def work(action):
    import torch
    from local_reproduce import acquire_gpu_queue
    from train_detail_formal import TrainingControl, TrainingPaused
    from utils.main_utils import read_text
    suite = load_suite()
    CONTROL.mkdir(parents=True, exist_ok=True)
    pause = CONTROL/'PAUSE.request'
    path = CONTROL/'queue.json'
    if action == 'resume':
        pause.unlink(missing_ok=True)
    elif pause.exists():
        raise ValueError('Pause requested; use Resume')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    queue = maybe(path) or dict(jobs={})

    def save():
        queue['updated_at'] = now()
        write_json(path, queue)

    # The GPU lock is shared with every other queue in this workspace. Wait instead of failing.
    while True:
        try:
            lock = acquire_gpu_queue()
            break
        except RuntimeError:
            queue.update(state='waiting_for_gpu', pid=os.getpid(), error=None)
            save()
            if pause.exists():
                queue['state'] = 'paused'
                save()
                return 75
            time.sleep(60)
    with lock:
        queue.update(state='running', pid=os.getpid(), error=None, started_at=queue.get('started_at', now()))
        child = None

        def job(key, function):
            record = queue['jobs'].setdefault(key, {})
            if record.get('status') == 'completed':
                return
            if pause.exists():
                raise TrainingPaused()
            queue['current_job'] = key
            record['uninterrupted'] = 'started_at' not in record
            record.update(status='running', started_at=record.get('started_at', now()))
            save()
            start = time.perf_counter()
            try:
                function()
            finally:
                record['active_seconds'] = record.get('active_seconds', 0)+time.perf_counter()-start
                save()
            record.update(status='completed', completed_at=now())
            save()
            print('COMPLETE', key, flush=True)

        save()
        try:
            queue['current_job'] = 'lock'
            save()
            plan = lock_plan(suite)
            queue['plan_digest'] = plan['digest']

            def check():
                from .check import run_check
                run_check(suite)
            job('check', check)
            from .evaluate import load_model, inspect_checkpoint, evaluate_cohort

            def train(run):
                nonlocal child
                if queue.get('child_pid') and process_alive(queue['child_pid']):
                    raise RuntimeError(f"A trainer from an earlier session is still running (PID {queue['child_pid']})")
                cfg = make_config(suite, run)
                free = shutil.disk_usage(ROOT).free/2**30
                if free < MIN_FREE_GIB:
                    raise RuntimeError(f'Only {free:.1f} GiB free on the drive; at least {MIN_FREE_GIB} GiB is needed')
                args = [sys.executable, '-u', 'train_hires.py', '--config-file', str(config_path(run)), '--output-dir', cfg.output_dir,
                        '--seed', str(cfg.seed), '--data_percentage', '100', '--num-workers', '0', '--control-dir', str(CONTROL),
                        '--gradient-checkpointing']
                if checkpoint_path(suite, run).exists():
                    args.append('--resume')
                logs = OUTPUT/'logs'
                logs.mkdir(parents=True, exist_ok=True)
                with (logs/f'{run}.log').open('a', encoding='utf-8') as stream:
                    child = subprocess.Popen(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    queue['child_pid'] = child.pid
                    save()
                    code = child.wait()
                child = None
                queue.pop('child_pid', None)
                if code == 75:
                    raise TrainingPaused()
                if code:
                    raise RuntimeError(f'{run} exited with code {code}; see outputs_research/{NAME}/logs/{run}.log')
                saved, _, digest = inspect_checkpoint(suite, run)
                queue['jobs']['train/'+run].update(checkpoint_sha256=digest, initial_fingerprints=saved.get('initial_fingerprints'))
                del saved

            def validate(run):
                model, cfg, digest = load_model(suite, run, queue['jobs']['train/'+run]['checkpoint_sha256'])
                control = TrainingControl(str(CONTROL), run_id=run, phase='evaluation')
                listing = Path(cfg.DATASET.TEXT_PROMPT_PATH)/'Val_text.xlsx'
                signature = dict(suite=NAME, run=run, checkpoint_sha256=digest, science_digest=plan['digest'], prompt=suite['prompt'],
                                 cohort='Val', list_sha256=sha256(listing), mc_samples=30, batch_size=32, threshold=.5)
                try:
                    summary = evaluate_cohort(model, cfg, suite['prompt'], suite['small_target_threshold'], 'Val', cfg.DATASET.VAL_PATH,
                                              read_text(listing), OUTPUT/'evaluations'/run/'Val', signature, control,
                                              save_maps=False, locked=False)
                    if summary['n'] != 100 or summary['n_small'] != 25:
                        raise ValueError('Unexpected Val cohort')
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

            for run in suite['runs']:
                job('train/'+run, lambda r=run: train(r))
                job('val/'+run, lambda r=run: validate(r))

            def report():
                from .report import write_report
                write_report()
            job('report', report)
            queue.update(state='completed', completed_at=now(), current_job=None)
            save()
            print('ALL COMPLETE. Report:', REPORT/'report.md', flush=True)
            return 0
        except TrainingPaused:
            queue['state'] = 'paused'
            current = queue.get('current_job')
            if current in queue['jobs'] and queue['jobs'][current].get('status') != 'completed':
                queue['jobs'][current]['status'] = 'paused'
            save()
            print('PAUSED. Use ResearchHires.cmd -Action Resume.', flush=True)
            return 75
        except BaseException as exc:
            if child is not None and child.poll() is None:
                pause.write_text(now(), encoding='utf-8')
                child.wait()
                pause.unlink(missing_ok=True)
            queue.update(state='failed', error=f'{type(exc).__name__}: {exc}')
            current = queue.get('current_job')
            if current in queue['jobs'] and queue['jobs'][current].get('status') != 'completed':
                queue['jobs'][current]['status'] = 'failed'
            save()
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['run', 'resume', 'pause', 'status', 'report'], default='status')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'status':
        status()
        return 0
    if args.action == 'pause':
        CONTROL.mkdir(parents=True, exist_ok=True)
        (CONTROL/'PAUSE.request').write_text(now(), encoding='utf-8')
        print('Pause requested; wait for "Queue: paused" before powering off.')
        return 0
    if args.action == 'report':
        from .report import write_report
        print(write_report())
        return 0
    return work(args.action)


if __name__ == '__main__':
    raise SystemExit(main())
