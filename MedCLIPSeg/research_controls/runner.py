"""Queue of the control experiments. Arm by arm: train three seeds, then evaluate that arm on Val and the test sets."""
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
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .common import *

MIN_FREE_GIB = 6
OTHER_QUEUES = [WORKSPACE/'control/research_detail_ext/queue.json',
                *(WORKSPACE/'control/research_zoom_ext'/name/'queue.json' for name in ('zoom_busi_v1', 'zoom_ablation_kvasir_v1')),
                WORKSPACE/'control/research_zoom_export/queue.json']


def process_alive(pid):
    try:
        if os.name == 'nt':
            listing = subprocess.run(['tasklist', '/FI', f'PID eq {int(pid)}', '/NH'], capture_output=True, text=True).stdout.lower()
            return str(int(pid)) in listing and 'python' in listing
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def others_waiting():
    """Earlier queues go first: the unfinished training queue, and any queue that is itself waiting for the GPU."""
    for path in OTHER_QUEUES:
        queue = maybe(path)
        state = queue.get('state')
        if path.parent.name == 'research_detail_ext' and state in ('running', 'paused', 'failed'):
            return str(path.parent.name)
        if state in ('waiting_for_gpu', 'waiting_for_training_queue'):
            age = (datetime.now(timezone.utc)-datetime.fromisoformat(queue['updated_at'])).total_seconds()
            if age < 300:
                return str(path.parent.name)
    return None


def job_keys(suite):
    keys = ['check']
    for arm in suite['arms']:
        runs = [f'{arm}_seed{seed}' for seed in suite['seeds']]
        keys += [f'train/{run}' for run in runs]
        keys += [f'eval/{run}/{cohort}' for run in runs for cohort in ['Val']+suite['test_targets']]
    return keys+['report']


def lock_plan(suite):
    from research_detail_formal.common import code_manifest as accepted_code
    from research_zoom_test.common import load_protocol as two_pass_protocol
    configs = {run: make_config(suite, run) for run in suite['runs']}
    two = two_pass_protocol()
    science = dict(suite=suite, protocol_sha256=sha256(SUITE_FILE), configs={r: science_config(c) for r, c in configs.items()},
                   accepted_training_code=accepted_code(), control_code=code_manifest(),
                   two_pass=dict(config=two['config'], zoom=two['zoom']))
    plan = dict(science, digest=object_hash(science))
    path = OUTPUT/'experiment_plan.json'
    if path.exists() and read(path)['digest'] != plan['digest'] and any(checkpoint_path(suite, r).exists() for r in suite['runs']):
        raise ValueError('Protocol or experiment code changed after training began; use a new suite name')
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
    if state == 'waiting_for_others':
        print('Waiting: an earlier queue has not finished (' + str(queue.get('waiting_for')) + '). Starts by itself.')
    if state == 'waiting_for_gpu':
        print('Waiting: another queue holds the GPU. Starts by itself.')
    cohorts = ['Val']+suite['test_targets']
    for arm in suite['arms']:
        runs = [f'{arm}_seed{seed}' for seed in suite['seeds']]
        trained = sum(done(f'train/{r}') for r in runs)
        evaluated = sum(done(f'eval/{r}/{c}') for r in runs for c in cohorts)
        seconds = [jobs[f'train/{r}']['active_seconds'] for r in runs if done(f'train/{r}')]
        note = f" | mean training time {sum(seconds)/len(seconds)/3600:.2f} h" if seconds else ''
        print(f'  {arm:16s} trained {trained}/3 | evaluations {evaluated}/{3*len(cohorts)}{note}')
    current = queue.get('current_job')
    if state == 'running' and current:
        line = f'Current: {current}'
        if current.startswith('train/') and progress.get('run_id') == current[6:] and progress.get('epoch'):
            line += f" | epoch {progress['epoch']}/{suite['epochs']}, {progress.get('phase')} batch {progress.get('step')}/{progress.get('total_steps')}"
            if progress.get('validation_dice') is not None:
                line += f" | last epoch Val Dice {progress['validation_dice']:.4f}"
        print(line)
    if queue.get('error'):
        print('ERROR:', queue['error'])
    if (REPORT/'report.md').exists():
        print('Report (updated after every arm):', REPORT/'report.md')
    print('Updated:', queue.get('updated_at', 'n/a'), '(UTC)')
    return state


def work(action):
    import torch
    from local_reproduce import acquire_gpu_queue
    from train_detail_formal import TrainingControl, TrainingPaused
    from utils.main_utils import read_text
    suite = load_suite()
    CONTROL.mkdir(parents=True, exist_ok=True)
    pause, path = CONTROL/'PAUSE.request', CONTROL/'queue.json'
    pause.unlink(missing_ok=True)   # starting again clears an earlier pause request
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    queue = maybe(path) or dict(jobs={})
    if queue.get('state') == 'completed':
        print('Already completed.')
        return 0

    def save():
        queue['updated_at'] = now()
        write_json(path, queue)

    def wait(state, **fields):
        queue.update(state=state, pid=os.getpid(), error=None, **fields)
        save()
        if pause.exists():
            queue['state'] = 'paused'
            save()
            return True
        time.sleep(60)
        return False

    while True:
        other = None if action == 'now' else others_waiting()
        if other:
            if wait('waiting_for_others', waiting_for=other):
                return 75
            continue
        try:
            lock = acquire_gpu_queue()
            break
        except RuntimeError:
            if wait('waiting_for_gpu'):
                return 75
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
            from .evaluate import load_model, inspect_checkpoint, upsampled_dataset, run_check
            job('check', lambda: write_json(CONTROL/'check.json', run_check(suite)))
            import research_zoom.evaluate as zoom_evaluate
            from .report import write_report
            standard_dataset = zoom_evaluate.dataset

            def train(run):
                nonlocal child
                if queue.get('child_pid') and process_alive(queue['child_pid']):
                    raise RuntimeError(f"A trainer from an earlier session is still running (PID {queue['child_pid']}); "
                                       'request Pause, wait for it to exit, then start again')
                cfg = make_config(suite, run)
                free = shutil.disk_usage(ROOT).free/2**30
                if free < MIN_FREE_GIB:
                    raise RuntimeError(f'Only {free:.1f} GiB free on the drive; at least {MIN_FREE_GIB} GiB is needed')
                args = [sys.executable, '-u', 'train_controls.py', '--config-file', str(config_path(run)), '--output-dir', cfg.output_dir,
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

            def evaluate_run(run):
                arm = split_run(suite, run)[0]
                spec = suite['arms'][arm]
                cohorts = ['Val']+suite['test_targets']
                if all(queue['jobs'].get(f'eval/{run}/{c}', {}).get('status') == 'completed' for c in cohorts):
                    return
                if pause.exists():
                    raise TrainingPaused()
                queue['current_job'] = f'load/{run}'
                save()
                model, cfg, digest = load_model(suite, run, queue['jobs']['train/'+run]['checkpoint_sha256'])
                tracker = TrainingControl(str(CONTROL), run_id=run, phase='evaluation')
                zoom_evaluate.dataset = upsampled_dataset if spec['data'] == 'upsampled_detail' else standard_dataset
                two_pass = dict(configs=[plan['two_pass']['config']], zoom=plan['two_pass']['zoom'])
                single = dict(configs=[], zoom=plan['two_pass']['zoom'])

                def evaluate(cohort):
                    if cohort == 'Val':
                        folder, listing, method = Path(cfg.DATASET.VAL_PATH), Path(cfg.DATASET.TEXT_PROMPT_PATH)/'Val_text.xlsx', single
                    else:
                        folder = ROOT/'data'/cohort/'Test_Folder'
                        listing = ROOT/'data'/cohort/'Prompts_Folder'/suite['test_list']
                        method = two_pass if spec['two_pass'] else single
                    zoom_evaluate.evaluate_condition(
                        model, cfg, folder, read_text(listing), method, OUTPUT/'evaluations'/run/cohort,
                        dict(suite=NAME, run=run, cohort=cohort, checkpoint_sha256=digest, science_digest=plan['digest'],
                             list_sha256=sha256(listing), configs=[c['name'] for c in method['configs']]), tracker, cohort)

                try:
                    for cohort in cohorts:
                        job(f'eval/{run}/{cohort}', lambda c=cohort: evaluate(c))
                    tracker.update('completed')
                finally:
                    zoom_evaluate.dataset = standard_dataset
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

            for arm in suite['arms']:
                runs = [f'{arm}_seed{seed}' for seed in suite['seeds']]
                for run in runs:
                    job('train/'+run, lambda r=run: train(r))
                for run in runs:
                    evaluate_run(run)
                write_report()     # partial report after every completed arm
                print('ARM COMPLETE', arm, '| report updated', flush=True)
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
            print('PAUSED. Start again with Start-Controls.cmd.', flush=True)
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
    parser.add_argument('--action', choices=['run', 'now', 'pause', 'status', 'report'], default='status')
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
