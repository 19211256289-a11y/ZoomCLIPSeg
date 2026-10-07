"""Queue of the reduced-training-set experiments. For each fraction (the smaller first) and each arm: train three seeds
with the accepted trainer, then evaluate them on the five polyp test sets (single pass and the locked SCZ).
Shares the GPU lock with every other queue; can be paused and continued."""
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

from .common import *

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
    """Seals protocol, configurations, training code, the reduced lists and the two-pass configuration.
    Computes no performance number."""
    from research_detail_formal.common import code_manifest as accepted_code
    from research_zoom_test.common import load_protocol as two_pass_protocol
    from .evaluate import inspect_lists
    configs = {run: make_config(suite, run) for run in suite['runs']}
    two = two_pass_protocol()
    science = dict(suite=suite, protocol_sha256=sha256(SUITE_FILE), configs={r: science_config(c) for r, c in configs.items()},
                   accepted_training_code=accepted_code(), control_code=code_manifest(), lists=inspect_lists(suite),
                   two_pass=dict(config=two['config'], zoom=two['zoom']))
    plan = dict(science, digest=object_hash(science))
    path = OUTPUT/'experiment_plan.json'
    if path.exists() and read(path)['digest'] != plan['digest'] and any(checkpoint_path(suite, r).exists() for r in suite['runs']):
        raise ValueError('Protocol, lists or experiment code changed after training began; use a new suite name')
    write_json(path, plan)
    for run, cfg in configs.items():
        config_path(run).parent.mkdir(parents=True, exist_ok=True)
        config_path(run).write_text(yaml.safe_dump(science_config(cfg), sort_keys=False), encoding='utf-8')
    return plan


def status():
    suite = load_suite()
    queue, progress = maybe(CONTROL/'queue.json'), maybe(CONTROL/'trainer_status.json')
    jobs, state = queue.get('jobs', {}), queue.get('state', 'not_started')
    done = lambda key: jobs.get(key, {}).get('status') == 'completed'
    print(f'{NAME}: {state} | pause requested: {(CONTROL/"PAUSE.request").exists()}')
    if state == 'waiting_for_gpu':
        print('    waiting: another queue holds the GPU; this queue starts by itself when the GPU is free')
    targets = suite['test_targets']
    for fraction, arm, runs in groups(suite):
        trained = sum(done(f'train/{r}') for r in runs)
        evaluated = sum(done(f'eval/{r}/{t}') for r in runs for t in targets)
        seconds = [jobs[f'train/{r}']['active_seconds'] for r in runs if done(f'train/{r}')]
        note = f' | mean training time {sum(seconds)/len(seconds)/60:.0f} min' if seconds else ''
        print(f'    {fraction:3d}% {arm:9s} trained {trained}/{len(runs)} | evaluations {evaluated}/{len(runs)*len(targets)}{note}')
    current = queue.get('current_job')
    if state == 'running' and current:
        line = f'    current: {current}'
        if current.startswith('train/') and progress.get('run_id') == current[6:] and progress.get('epoch'):
            line += f" | epoch {progress['epoch']}/{suite['epochs']}"
            if progress.get('validation_dice') is not None:
                line += f", last epoch Val Dice {progress['validation_dice']:.4f}"
        print(line)
    if queue.get('error'):
        print('    ERROR:', queue['error'])
    if (REPORT/'report.md').exists():
        print('    report (updated after every arm):', REPORT/'report.md')
    print('    updated:', queue.get('updated_at', 'n/a'), '(UTC)')
    return state


def work():
    import torch
    from local_reproduce import acquire_gpu_queue
    from train_detail_formal import TrainingControl, TrainingPaused
    from utils.main_utils import read_text
    suite = load_suite()
    CONTROL.mkdir(parents=True, exist_ok=True)
    pause, path = CONTROL/'PAUSE.request', CONTROL/'queue.json'
    pause.unlink(missing_ok=True)   # starting again clears an earlier pause request
    queue = maybe(path) or dict(jobs={})
    if queue.get('state') == 'completed':
        print(NAME, 'already completed. Report:', REPORT/'report.md', flush=True)
        return 0
    queue.update(n_jobs=len(job_keys(suite)), error=None, pid=os.getpid())

    def save():
        queue['updated_at'] = now()
        write_json(path, queue)

    announced = False
    while True:
        try:
            lock = acquire_gpu_queue()
            break
        except RuntimeError:
            queue['state'] = 'waiting_for_gpu'
            save()
            if not announced:
                print('Waiting for the GPU (another queue is running). This starts by itself when the GPU is free ...', flush=True)
                announced = True
            if pause.exists():
                queue['state'] = 'paused'
                save()
                return 75
            time.sleep(30)
    with lock:
        queue.update(state='running', started_at=queue.get('started_at', now()))
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
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is not available')
            queue['current_job'] = 'lock'
            save()
            print('Verifying the reduced training lists and sealing the protocol ...', flush=True)
            plan = lock_plan(suite)
            if queue.get('plan_digest') not in (None, plan['digest']):
                queue['jobs'] = {}   # only reachable before any checkpoint of this suite exists (see lock_plan)
            queue['plan_digest'] = plan['digest']
            save()
            from .evaluate import load_model, inspect_checkpoint, run_check
            import research_zoom.evaluate as zoom_evaluate
            from .report import write_report
            job('check', lambda: write_json(CONTROL/'check.json', dict(run_check(suite), checked_at=now())))
            two_pass = dict(configs=[plan['two_pass']['config']], zoom=plan['two_pass']['zoom'])

            def train(run):
                nonlocal child
                if queue.get('child_pid') and process_alive(queue['child_pid']):
                    raise RuntimeError(f"A trainer from an earlier session is still running (PID {queue['child_pid']}); "
                                       'request Pause, wait for it to exit, then start again')
                cfg = make_config(suite, run)
                fraction = split_run(suite, run)[1]
                free = shutil.disk_usage(ROOT).free/2**30
                if free < MIN_FREE_GIB:
                    raise RuntimeError(f'Only {free:.1f} GiB free on the drive; at least {MIN_FREE_GIB} GiB is needed')
                args = [sys.executable, '-u', 'train_controls.py', '--config-file', str(config_path(run)), '--output-dir', cfg.output_dir,
                        '--seed', str(cfg.seed), '--data_percentage', str(fraction), '--num-workers', '0', '--control-dir', str(CONTROL),
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
                saved, _, digest = inspect_checkpoint(suite, run)     # also verifies the list the trainer sealed
                queue['jobs']['train/'+run].update(checkpoint_sha256=digest, initial_fingerprints=saved.get('initial_fingerprints'),
                                                   training_images=plan['lists'][f'p{fraction}']['Train']['n'])
                del saved

            def evaluate_run(run):
                targets = suite['test_targets']
                if all(queue['jobs'].get(f'eval/{run}/{t}', {}).get('status') == 'completed' for t in targets):
                    return
                if pause.exists():
                    raise TrainingPaused()
                queue['current_job'] = f'load/{run}'
                save()
                model, cfg, digest = load_model(suite, run, queue['jobs']['train/'+run]['checkpoint_sha256'])
                tracker = TrainingControl(str(CONTROL), run_id=run, phase='evaluation')

                def evaluate(target):
                    folder = ROOT/'data'/target/'Test_Folder'
                    listing = ROOT/'data'/target/'Prompts_Folder'/suite['test_list']
                    zoom_evaluate.evaluate_condition(
                        model, cfg, folder, read_text(listing), two_pass, OUTPUT/'evaluations'/run/target,
                        dict(suite=NAME, run=run, cohort=target, checkpoint_sha256=digest, science_digest=plan['digest'],
                             list_sha256=sha256(listing), configs=[c['name'] for c in two_pass['configs']]), tracker, target)

                try:
                    for target in targets:
                        job(f'eval/{run}/{target}', lambda t=target: evaluate(t))
                    tracker.update('completed')
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

            for fraction, arm, runs in groups(suite):
                keys = [f'train/{r}' for r in runs]+[f'eval/{r}/{t}' for r in runs for t in suite['test_targets']]
                if all(queue['jobs'].get(k, {}).get('status') == 'completed' for k in keys):
                    continue
                for run in runs:
                    job('train/'+run, lambda r=run: train(r))
                for run in runs:
                    evaluate_run(run)
                write_report(partial=True)     # the report is refreshed after every completed arm
                print(f'ARM COMPLETE {arm} at {fraction}% | report updated', flush=True)
            job('report', write_report)
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
            print('PAUSED. Start again with the same command; completed epochs and jobs are kept.', flush=True)
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
    parser.add_argument('--action', choices=['run', 'pause', 'status', 'report'], default='status')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'status':
        status()
        return 0
    if args.action == 'pause':
        CONTROL.mkdir(parents=True, exist_ok=True)
        (CONTROL/'PAUSE.request').write_text(now(), encoding='utf-8')
        print('Pause requested; wait until the state is no longer "running" before powering off.')
        return 0
    if args.action == 'report':
        from .report import write_report
        print(write_report(partial=maybe(CONTROL/'queue.json').get('state') != 'completed'))
        return 0
    return work()


if __name__ == '__main__':
    raise SystemExit(main())
