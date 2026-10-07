"""One queue for both extension suites: lock, train, Val, then Test/OOD and report per suite."""
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

BATCHES = dict(BUSI=26+4, Kvasir=34+5)   # train + validation batches per epoch at batch 24
MIN_FREE_GIB = 6


def process_alive(pid):
    """True if a Python process with this PID exists. Never signals the process."""
    try:
        if os.name == 'nt':
            listing = subprocess.run(['tasklist', '/FI', f'PID eq {int(pid)}', '/NH'], capture_output=True, text=True).stdout.lower()
            return str(int(pid)) in listing and 'python' in listing
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def assignment():
    """Which suites this machine runs. Without the file, the machine runs both."""
    value = maybe(CONTROL/'assignment.json')
    suites = value.get('suites', list(SUITES))
    if not suites or any(name not in SUITES for name in suites):
        raise ValueError('Invalid control/research_detail_ext/assignment.json')
    return dict(suites=suites, report=value.get('report', True), host=value.get('host', 'local'))


def lock_plan(suite):
    """Seal the scientific protocol before any training of this suite. Computes no performance number."""
    from research_formal.metrics import foreground_threshold
    from research_detail_formal.common import code_manifest as accepted_code
    if not read(PARENT_AUDIT)['passed']:
        raise ValueError('Parent acceptance did not pass')
    configs = {run: make_config(suite, run) for run in suite['runs']}
    first = configs[suite['runs'][0]]
    threshold = foreground_threshold(first)
    if 'small_target_threshold' in suite and threshold != suite['small_target_threshold']:
        raise ValueError('Small-target threshold differs from the locked value')
    for key, _ in FOLDERS:
        if not Path(first.DATASET[key]).is_dir():
            raise FileNotFoundError(first.DATASET[key])
    science = dict(suite=suite, protocol_sha256=sha256(suite_file(suite['name'])),
                   configs={run: science_config(cfg) for run, cfg in configs.items()},
                   accepted_training_code=accepted_code(), trainer_entry_sha256=sha256(ROOT/'train_detail_ext.py'),
                   base_weights_sha256=sha256(first.RESEARCH.BASE_WEIGHTS), small_threshold=threshold,
                   test_cohorts=cohort_manifest(suite, first))
    plan = dict(science, digest=object_hash(science), tooling=tooling_manifest())
    path = output(suite)/'experiment_plan.json'
    if path.exists():
        previous = read(path)
        if previous['digest'] != plan['digest'] and any(checkpoint_path(suite, run).exists() for run in suite['runs']):
            raise ValueError(f"{suite['name']}: protocol, configuration or data changed after training began; use a new suite name")
        if previous['tooling'] != plan['tooling'] and any(
                maybe(output(suite)/'evaluations'/run/target/'manifest.json').get('complete')
                for run in suite['runs'] for target in suite['test_targets']):
            raise ValueError(f"{suite['name']}: evaluation code changed after Test results were produced")
    write_json(path, plan)
    for run, cfg in configs.items():
        config_path(suite, run).parent.mkdir(parents=True, exist_ok=True)
        config_path(suite, run).write_text(yaml.safe_dump(science_config(cfg), sort_keys=False), encoding='utf-8')
    return plan


def smoke(suite):
    """Load the first and last item of every evaluation list on CPU, so list or file problems surface before training."""
    from utils.main_utils import read_text
    from research.data import DualScaleDataset, DualScaleTransform
    cfg = make_config(suite, suite['runs'][0])
    cohorts = [('Val', Path(cfg.DATASET.VAL_PATH), Path(cfg.DATASET.TEXT_PROMPT_PATH)/'Val_text.xlsx')]
    cohorts += [(t, ROOT/'data'/t/'Test_Folder', ROOT/'data'/t/'Prompts_Folder'/suite['test_list']) for t in suite['test_targets']]
    for cohort, folder, listing in cohorts:
        data = DualScaleDataset(str(folder), cohort, read_text(listing), DualScaleTransform(False),
                                image_size=224, detail_size=448, upsampled=False)
        for index in (0, len(data)-1):
            sample = data[index]
            if tuple(sample['image'].shape) != (3, 224, 224) or tuple(sample['image_detail'].shape) != (3, 448, 448):
                raise ValueError(f'Unexpected tensor shape in {cohort}')
            if not (folder/'label'/sample['mask_name']).is_file():
                raise FileNotFoundError(folder/'label'/sample['mask_name'])


def job_keys(suite):
    name = suite['name']
    keys = []
    for run in suite['runs']:
        keys += [f'{name}/train/{run}', f'{name}/val/{run}']
    keys += [f'{name}/test/{run}/{target}' for run in suite['runs'] for target in suite['test_targets']]
    return keys+[f'{name}/report']


def status():
    queue, progress = maybe(CONTROL/'queue.json'), maybe(CONTROL/'trainer_status.json')
    jobs = queue.get('jobs', {})
    done = lambda key: jobs.get(key, {}).get('status') == 'completed'
    state = queue.get('state', 'not_started')
    print('Queue:', state, '| Pause requested:', (CONTROL/'PAUSE.request').exists())
    current = queue.get('current_job')
    rates = [jobs[k]['active_seconds']/(100*BATCHES[load_suite(k.split('/')[0])['dataset']])
             for k in jobs if '/train/' in k and done(k) and jobs[k].get('uninterrupted')]
    rate = sum(rates)/len(rates) if rates else 1.7
    remaining = 0.
    mine = assignment()
    print('Host:', mine['host'], '| suites on this machine:', ', '.join(mine['suites']))
    for name in mine['suites']:
        suite = load_suite(name)
        runs = suite['runs']
        trained = sum(done(f'{name}/train/{r}') for r in runs)
        tested = sum(done(f'{name}/test/{r}/{t}') for r in runs for t in suite['test_targets'])
        print(f"\n{name} ({suite['dataset']}): trained {trained}/{len(runs)} | Val {sum(done(f'{name}/val/{r}') for r in runs)}/{len(runs)}"
              f" | Test {tested}/{len(runs)*len(suite['test_targets'])} | report {'done' if done(f'{name}/report') else ('pending' if mine['report'] else 'built on the PC after Fetch')}")
        for run in runs:
            value = maybe(output(suite)/'evaluations'/run/'Val/summary.json')
            if value:
                print(f"  {run}: Val DSC {value['DSC_percent']:.2f}, small {value['small_DSC_percent']:.2f}")
            if not done(f'{name}/train/{run}'):
                epochs = 100
                if current == f'{name}/train/{run}' and progress.get('run_id') == run and progress.get('epoch'):
                    epochs = 100-max(0, progress['epoch']-1)
                remaining += epochs*BATCHES[suite['dataset']]*rate
            remaining += 60*(not done(f'{name}/val/{run}'))+70*sum(not done(f'{name}/test/{run}/{t}') for t in suite['test_targets'])
    if current and state == 'running':
        line = f'\nCurrent: {current}'
        if progress.get('status') in ('running', 'starting') and progress.get('phase') in ('train', 'validation', 'epoch_complete'):
            line += f" | epoch {progress.get('epoch')}/100, {progress.get('phase')} batch {progress.get('step')}/{progress.get('total_steps')}"
            if progress.get('validation_dice') is not None:
                line += f" | last epoch Val Dice {progress['validation_dice']:.4f}"
        elif progress.get('phase') == 'evaluation':
            line += f" | batch {progress.get('step')}/{progress.get('total_steps')}"
        print(line)
    if state != 'completed':
        print(f'Estimated remaining: {remaining/3600:.1f} h (excludes pauses; refined as runs finish)')
    if queue.get('error'):
        print('ERROR:', queue['error'])
    print('Updated:', queue.get('updated_at', 'n/a'), '(UTC)')
    return state


def work(action):
    import torch
    from local_reproduce import acquire_gpu_queue
    mine = assignment()
    from train_detail_formal import TrainingControl, TrainingPaused
    from utils.main_utils import read_text
    CONTROL.mkdir(parents=True, exist_ok=True)
    pause = CONTROL/'PAUSE.request'
    if action == 'resume':
        pause.unlink(missing_ok=True)
    elif pause.exists():
        raise ValueError('Pause requested; use Resume')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    with acquire_gpu_queue():
        path = CONTROL/'queue.json'
        queue = maybe(path) or dict(jobs={})
        queue.update(state='running', pid=os.getpid(), error=None, started_at=queue.get('started_at', now()), host=mine['host'],
                     assigned_suites=mine['suites'])
        child = None

        def save():
            queue['updated_at'] = now()
            write_json(path, queue)

        def finished(key):
            return queue['jobs'].get(key, {}).get('status') == 'completed'

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
            suites = [load_suite(name) for name in mine['suites']]
            queue['current_job'] = 'lock'
            save()
            plans = {suite['name']: lock_plan(suite) for suite in suites}
            for suite in suites:
                smoke(suite)
            queue['plan_digests'] = {name: plan['digest'] for name, plan in plans.items()}
            save()
            from .evaluate import load_model, inspect_checkpoint, evaluate_cohort
            for suite in suites:
                name, plan = suite['name'], plans[suite['name']]
                threshold = plan['small_threshold']

                def train(run):
                    nonlocal child
                    if queue.get('child_pid') and process_alive(queue['child_pid']):
                        raise RuntimeError(f"A trainer from an earlier session is still running (PID {queue['child_pid']}); "
                                           'request Pause and wait for it to exit, then Resume')
                    cfg = make_config(suite, run)
                    free = shutil.disk_usage(ROOT).free/2**30
                    if free < MIN_FREE_GIB:
                        raise RuntimeError(f'Only {free:.1f} GiB free on the drive; at least {MIN_FREE_GIB} GiB is needed for the next run')
                    args = [sys.executable, '-u', 'train_detail_ext.py', '--config-file', str(config_path(suite, run)),
                            '--output-dir', cfg.output_dir, '--seed', str(cfg.seed), '--data_percentage', '100',
                            '--num-workers', '0', '--control-dir', str(CONTROL), '--gradient-checkpointing']
                    if checkpoint_path(suite, run).exists():
                        args.append('--resume')
                    logs = output(suite)/'logs'
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
                        raise RuntimeError(f'{run} exited with code {code}; see outputs_research/{name}/logs/{run}.log')
                    saved, _, digest = inspect_checkpoint(suite, run)
                    record = queue['jobs'][f'{name}/train/{run}']
                    record.update(checkpoint_sha256=digest, initial_fingerprints=saved.get('initial_fingerprints'),
                                  train_size=saved['runtime']['train_size'], val_size=saved['runtime']['val_size'])
                    del saved

                def signature(run, digest, cohort, listing):
                    return dict(suite=name, run=run, checkpoint_sha256=digest, science_digest=plan['digest'], prompt=suite['prompt'],
                                cohort=cohort, list_sha256=sha256(listing), mc_samples=30, batch_size=32, threshold=.5,
                                small_threshold=threshold)

                def validate(run):
                    model, cfg, digest = load_model(suite, run, queue['jobs'][f'{name}/train/{run}']['checkpoint_sha256'])
                    control = TrainingControl(str(CONTROL), run_id=run, suite=name, phase='evaluation')
                    listing = Path(cfg.DATASET.TEXT_PROMPT_PATH)/'Val_text.xlsx'
                    try:
                        evaluate_cohort(model, cfg, suite['prompt'], threshold, 'Val', cfg.DATASET.VAL_PATH, read_text(listing),
                                        output(suite)/'evaluations'/run/'Val', signature(run, digest, 'Val', listing), control,
                                        save_maps=False, locked=False)
                    finally:
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()

                for run in suite['runs']:
                    job(f'{name}/train/{run}', lambda r=run: train(r))
                    job(f'{name}/val/{run}', lambda r=run: validate(r))
                # No Test cohort is touched until every run of the suite has its final checkpoint and Val result.
                for run in suite['runs']:
                    keys = [f'{name}/test/{run}/{target}' for target in suite['test_targets']]
                    if all(finished(k) for k in keys):
                        continue
                    if pause.exists():
                        raise TrainingPaused()
                    queue['current_job'] = f'{name}/load/{run}'
                    save()
                    model, cfg, digest = load_model(suite, run, queue['jobs'][f'{name}/train/{run}']['checkpoint_sha256'])
                    control = TrainingControl(str(CONTROL), run_id=run, suite=name, phase='evaluation')

                    def test(target):
                        cohort = plan['test_cohorts'][target]
                        listing = ROOT/'data'/target/'Prompts_Folder'/suite['test_list']
                        if sha256(listing) != cohort['list_sha256']:
                            raise ValueError('Test list changed after locking')
                        evaluate_cohort(model, cfg, suite['prompt'], threshold, target, ROOT/'data'/target/'Test_Folder',
                                        read_text(listing), output(suite)/'evaluations'/run/target,
                                        dict(signature(run, digest, target, listing), files_digest=object_hash(cohort['files'])),
                                        control, groups=list(cohort['image_hash_groups'].values()))

                    try:
                        for target in suite['test_targets']:
                            job(f'{name}/test/{run}/{target}', lambda t=target: test(t))
                    finally:
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                if mine['report']:
                    def report():
                        from .report import write_report
                        write_report(name)
                    job(f'{name}/report', report)
            queue.update(state='completed', completed_at=now(), current_job=None)
            save()
            print('ALL COMPLETE', flush=True)
            return 0
        except TrainingPaused:
            queue['state'] = 'paused'
            current = queue.get('current_job')
            if current in queue['jobs'] and queue['jobs'][current].get('status') != 'completed':
                queue['jobs'][current]['status'] = 'paused'
            save()
            print('PAUSED. Use ResearchDetailExt.cmd -Action Resume.', flush=True)
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


def preflight():
    """Seal both protocols and print what will run; starts no training and reads no Test result."""
    for name in assignment()['suites']:
        suite = load_suite(name)
        plan = lock_plan(suite)
        smoke(suite)
        print(f"{name}: {len(suite['runs'])} runs, small-target threshold {plan['small_threshold']:.6f}")
        for target, cohort in plan['test_cohorts'].items():
            print(f"  {target}: {cohort['n']} records, {cohort['n_unique_images']} unique, overlap with source Train/Val "
                  f"{len(cohort['overlap_with_source_train_val'])}, shared {cohort['shared_unique_images_with_other_tests'] or 'none'}")
    print('PREFLIGHT OK')
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['preflight', 'run', 'resume', 'pause', 'status', 'report'], default='status')
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
    if args.action == 'preflight':
        return preflight()
    if args.action == 'report':
        from .report import write_report
        for name in SUITES:
            try:
                print(write_report(name))
            except (ValueError, FileNotFoundError) as exc:
                print(f'{name}: report not built yet ({exc})')
        return 0
    return work(args.action)


if __name__ == '__main__':
    raise SystemExit(main())
