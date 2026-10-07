"""Independent, single-GPU research queue. No action runs on import."""
from local_runtime import configure_local_models
configure_local_models()

import argparse
import json
import os
import subprocess
import sys
import time
from statistics import median
from pathlib import Path

import yaml

from .common import (ROOT, WORKSPACE, VARIANTS, TARGETS, now, write_json, object_hash,
                     load_suite, control_dir, suite_output, make_config, checkpoint_path,
                     plain_config, code_manifest, make_signature, assert_signature, test_cohort_manifest)


def read_state(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else (default or {})


def require_check(suite):
    result = read_state(control_dir(suite) / 'check.json')
    if not result.get('passed') or result.get('code_digest') != object_hash(code_manifest()):
        raise RuntimeError('Run ResearchFormal.cmd -Action Check for this suite first; validation is absent or stale.')
    return bool(result['gradient_checkpointing'])


def lock_plan(suite, checkpointing):
    path = suite_output(suite) / 'experiment_plan.json'
    plan = {'suite': suite, 'checkpointing': checkpointing, 'test_cohorts': test_cohort_manifest(suite),
            'signatures': {v: make_signature(make_config(suite, v, checkpointing)) for v in suite['runs']}}
    plan['digest'] = object_hash(plan)
    if path.exists() and read_state(path).get('digest') != plan['digest']:
        raise RuntimeError('Locked experiment plan changed. Use a new suite name for a new experiment.')
    if not path.exists():
        write_json(path, plan)
    return plan


def prepare(suite):
    """Lock configuration without creating a queue, checkpoint, or training process."""
    checkpointing = require_check(suite)
    plan = lock_plan(suite, checkpointing)
    directory = suite_output(suite)/'configs'
    directory.mkdir(parents=True, exist_ok=True)
    for run in suite['runs']:
        cfg = make_config(suite, run, checkpointing)
        (directory/f'{run}.yaml').write_text(yaml.safe_dump(
            {k: plain_config(cfg[k]) for k in ('DATASET', 'TRAIN', 'TEST', 'MODEL', 'RESEARCH')}, sort_keys=False), encoding='utf-8')
    print('READY: protocol and nine configurations locked; no formal training started.', flush=True)
    return plan


def final_evaluation_jobs(suite):
    return [(f'test/{run}/{target}', run, target) for run in suite['runs'] for target in suite['test_targets']]


def require_all_final_checkpoints(suite, checkpointing, plan):
    import torch
    for run in suite['runs']:
        cfg = make_config(suite, run, checkpointing)
        saved = torch.load(checkpoint_path(suite, run), map_location='cpu', weights_only=False)
        assert_signature(saved.get('research_signature'), plan['signatures'][run])
        if saved['epoch'] != cfg.TRAIN.NUM_EPOCHS-1 or saved['progress']['phase'] != 'epoch_complete':
            raise ValueError(f'{run} has not completed its final training epoch')
        del saved


def status(suite):
    directory = control_dir(suite)
    queue = read_state(directory / 'queue.json')
    progress = read_state(directory / 'trainer_status.json')
    print(f"Suite: {suite['name']} | Queue: {queue.get('state', 'not_started')} | Pause requested: {(directory/'PAUSE.request').exists()}")
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    for key, record in queue.get('jobs', {}).items():
        print(f"{key}: {record.get('status', 'unknown')}")
    completed = sum(queue.get('jobs', {}).get('val/'+run, {}).get('status') == 'completed' for run in suite['runs'])
    tested = sum(queue.get('jobs', {}).get(key, {}).get('status') == 'completed' for key, _, _ in final_evaluation_jobs(suite))
    print(f"Completed training/Val runs: {completed}/{len(suite['runs'])} | Test cohorts: {tested}/45 | Automatic tests after all nine runs")
    if not queue:
        ready = (suite_output(suite)/'experiment_plan.json').exists()
        print('Protocol prepared:', ready, '| Formal training has NOT started.')
    seconds_per_epoch = [job['active_seconds']/suite['epochs'] for key, job in queue.get('jobs', {}).items()
                         if key.startswith('train/') and job.get('status') == 'completed' and job.get('active_seconds')]
    current_key = queue.get('current_job', '')
    valid_progress = current_key == 'train/'+progress.get('run_id', '') and progress.get('phase') in ('train', 'validation', 'epoch_complete')
    if valid_progress and progress.get('remaining_seconds') and progress.get('epoch'):
        seconds_per_epoch.append(progress['remaining_seconds']/max(1, suite['epochs']-progress['epoch']+1))
    if seconds_per_epoch:
        rate = median(seconds_per_epoch)
        remaining = 0.0
        for run in suite['runs']:
            job = queue.get('jobs', {}).get('train/'+run, {})
            if job.get('status') != 'completed':
                epochs = suite['epochs']
                if valid_progress and current_key == 'train/'+run:
                    epochs -= max(0, progress['epoch']-1)
                    if progress['phase'] == 'train':
                        epochs -= progress.get('step', 0)/max(1, progress.get('total_steps', 34))
                    else:
                        epochs -= .9
                remaining += max(0, epochs)*rate
            if queue.get('jobs', {}).get('val/'+run, {}).get('status') != 'completed':
                remaining += 65
        plan = read_state(suite_output(suite)/'experiment_plan.json')
        per_image = [job['active_seconds']/plan['test_cohorts'][key.rsplit('/',1)[1]]['n']
                     for key, job in queue.get('jobs', {}).items()
                     if key.startswith('test/') and job.get('status') == 'completed' and job.get('active_seconds')]
        evaluation_rate = median(per_image) if per_image else .75
        for key, _, target in final_evaluation_jobs(suite):
            if queue.get('jobs', {}).get(key, {}).get('status') != 'completed':
                remaining += plan.get('test_cohorts', {}).get(target, {}).get('n', 100)*evaluation_rate
        print(f"Estimated active time remaining incl. automatic tests: {remaining/3600:.2f} hours (excludes pauses)")
    if queue.get('error'):
        print('ERROR:', queue['error'])
    print('Output:', suite_output(suite))
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=5)
        print(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass


def work(suite, action):
    from train_formal import TrainingPaused
    from local_reproduce import acquire_gpu_queue
    directory = control_dir(suite)
    directory.mkdir(parents=True, exist_ok=True)
    queue_path = directory / 'queue.json'
    # Acquire the same OS lock used by the old reproduction coordinator.
    with acquire_gpu_queue():
        if action == 'check':
            from .check import run_checks
            run_checks(suite)
            prepare(suite)
            return 0
        checkpointing = require_check(suite)
        previous = read_state(queue_path)
        if action == 'resume':
            if previous.get('state') not in ('paused', 'failed', 'running'):
                raise ValueError('No interrupted research action to resume')
            action = previous['action']
        elif previous.get('state') in ('paused', 'failed', 'running') and previous.get('action') != action:
            raise ValueError('Resume the existing research action before changing the action')
        plan = lock_plan(suite, checkpointing)
        if previous.get('plan_digest') and previous['plan_digest'] != plan['digest']:
            raise ValueError('Queue and locked experiment plan disagree')
        (directory / 'PAUSE.request').unlink(missing_ok=True)
        queue = previous or {'jobs': {}}
        queue.update(state='running', action=action, pid=os.getpid(), plan_digest=plan['digest'], updated_at=now(), error=None)
        write_json(queue_path, queue)
        child = None

        def save():
            queue['updated_at'] = now()
            write_json(queue_path, queue)

        def run_job(key, function):
            if (directory / 'PAUSE.request').exists():
                raise TrainingPaused()
            job = queue['jobs'].setdefault(key, {})
            if job.get('status') == 'completed':
                print('SKIP completed', key, flush=True)
                return
            job.update(status='running', started_at=job.get('started_at', now()))
            queue['current_job'] = key
            save()
            print('START', key, flush=True)
            started_clock = time.perf_counter()
            try:
                function()
            finally:
                job['active_seconds'] = job.get('active_seconds', 0.0) + time.perf_counter()-started_clock
                save()
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            job.update(status='completed', completed_at=now())
            save()
            print('COMPLETE', key, flush=True)
            if key.startswith(('val/', 'test/')):
                from .report import write_report
                write_report(suite)

        def train_variant(variant):
            nonlocal child
            cfg = make_config(suite, variant, checkpointing)
            cfg_dir = suite_output(suite) / 'configs'
            cfg_dir.mkdir(parents=True, exist_ok=True)
            cfg_file = cfg_dir / f'{variant}.yaml'
            cfg_file.write_text(yaml.safe_dump({k: plain_config(cfg[k]) for k in ('DATASET', 'TRAIN', 'TEST', 'MODEL', 'RESEARCH')}, sort_keys=False), encoding='utf-8')
            args = [sys.executable, '-u', 'train_formal.py', '--config-file', str(cfg_file), '--output-dir', cfg.output_dir,
                    '--seed', str(cfg.seed), '--data_percentage', '100', '--num-workers', '0', '--control-dir', str(directory)]
            if checkpointing:
                args.append('--gradient-checkpointing')
            if checkpoint_path(suite, variant).exists():
                args.append('--resume')
            logs = suite_output(suite) / 'logs'
            logs.mkdir(exist_ok=True)
            with (logs / f'{variant}.log').open('a', encoding='utf-8') as stream:
                child = subprocess.Popen(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                queue['child_pid'] = child.pid
                save()
                code = child.wait()
            child = None
            queue.pop('child_pid', None)
            if code == 75:
                raise TrainingPaused()
            if code:
                raise RuntimeError(f'{variant} training exited {code}; see {logs / (variant+".log")}')

        def evaluate_variant(variant, split, target):
            from .evaluate import evaluate
            cfg = make_config(suite, variant, checkpointing)
            return evaluate(cfg, checkpoint_path(suite, variant),
                            suite_output(suite) / 'evaluations' / variant / split / target,
                            directory, target, split)

        try:
            if action == 'train':
                for variant in suite['runs']:
                    run_job('train/'+variant, lambda v=variant: train_variant(v))
                    run_job('val/'+variant, lambda v=variant: evaluate_variant(v, 'Val', suite['dataset']))
            elif action == 'evaluate':
                require_all_final_checkpoints(suite, checkpointing, plan)
                for variant in suite['runs']:
                    run_job('val/'+variant, lambda v=variant: evaluate_variant(v, 'Val', suite['dataset']))
            else:
                raise ValueError(action)
            # No test is run until ALL nine final checkpoints exist and match the locked protocol.
            require_all_final_checkpoints(suite, checkpointing, plan)
            for key, variant, target in final_evaluation_jobs(suite):
                run_job(key, lambda v=variant, t=target: evaluate_variant(v, 'Test', t))
            from .report import write_report
            print('Report:', write_report(suite), flush=True)
            queue.update(state='completed', completed_at=now())
            save()
            return 0
        except TrainingPaused:
            queue['state'] = 'paused'
            if queue.get('current_job'):
                queue['jobs'][queue['current_job']]['status'] = 'paused'
            save()
            print('PAUSED: checkpoint preserved, GPU queue released. Use ResearchFormal.cmd -Action Resume.', flush=True)
            return 75
        except BaseException as exc:
            # Interrupting the coordinator requests a safe child pause, never leaves an orphan.
            if child is not None and child.poll() is None:
                (directory / 'PAUSE.request').write_text(now(), encoding='utf-8')
                child.wait()
            queue.update(state='failed', error=f'{type(exc).__name__}: {exc}')
            if queue.get('current_job'):
                queue['jobs'][queue['current_job']]['status'] = 'failed'
            save()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True, choices=['check', 'train', 'pause', 'resume', 'status', 'evaluate'])
    parser.add_argument('--suite', default='formal_kvasir_v1')
    args = parser.parse_args()
    os.chdir(ROOT)
    suite = load_suite(args.suite)
    directory = control_dir(suite)
    if args.action == 'pause':
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'PAUSE.request').write_text(now(), encoding='utf-8')
        print('Pause requested. Wait for status=paused and the coordinator to exit before powering off.')
        return 0
    if args.action == 'status':
        status(suite)
        return 0
    return work(suite, args.action)


if __name__ == '__main__':
    sys.exit(main())
