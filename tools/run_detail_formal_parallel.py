"""Two seed lanes, with unchanged signed training code and serial final evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / 'MedCLIPSeg'
HOST=os.environ.get('MEDCLIPSEG_HOST','local')
if HOST not in ('local','server4090'):raise ValueError('Unknown host')
DIRECTORY=WORKSPACE/'control/research_detail_formal/formal_detail_mix_kvasir_v1'/HOST
LANES=({0:['baseline_seed666','detail_seed666'],1:['global_mix_seed666','detail_global_mix_seed666']}
       if HOST=='local' else {s:[f'{m}_seed{s}' for m in ('baseline','global_mix','detail','detail_global_mix')] for s in (667,668)})
RUNS=[r for runs in LANES.values() for r in runs]
HANDOFF_PATH=DIRECTORY/'execution_assignment.json'
ASSIGNMENT=json.loads(HANDOFF_PATH.read_text()) if HANDOFF_PATH.exists() else None
EXECUTOR=os.environ.get('MEDCLIPSEG_EXECUTOR',HOST)
if ASSIGNMENT:
    expected=[f'{m}_seed{s}' for s in (667,668) for m in ('baseline','global_mix','detail','detail_global_mix')]
    assert HOST=='server4090' and sorted(ASSIGNMENT['owners'])==sorted(expected)
    assert EXECUTOR in ('server4090','server5090')
    assert all(v in ('server4090','server5090') for v in ASSIGNMENT['owners'].values())
    RUNS=[r for r in RUNS if ASSIGNMENT['owners'][r]==EXECUTOR]
    LANES=({0:RUNS} if EXECUTOR=='server5090' else {s:[r for r in rs if r in RUNS] for s,rs in LANES.items()})


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def pause_all(directory=DIRECTORY):
    directory.mkdir(parents=True, exist_ok=True)
    # Parent marker comes first: the coordinator cannot launch further work.
    (directory / 'PAUSE.request').write_text('manual or coordinated pause', encoding='utf-8')
    for run in RUNS:
        child_dir = directory / 'workers' / run
        child_dir.mkdir(parents=True, exist_ok=True)
        (child_dir / 'PAUSE.request').write_text('pause both seed lanes', encoding='utf-8')


def schedule(lanes, jobs, launch, verify, save, is_paused, signal_pause, sleep=time.sleep):
    """At most one active child per seed; pause/failure drains all children safely."""
    active = {}
    error = None
    stopping = False

    def drain_record(run, code, elapsed):
        nonlocal error, stopping
        record = jobs['train/' + run]
        record['active_seconds'] = record.get('active_seconds', 0.) + elapsed
        record.pop('pid', None)
        if code == 0:
            try:
                record['checkpoint_sha256'] = verify(run)
                record['status'] = 'completed'
                from datetime import datetime, timezone
                record['completed_at'] = datetime.now(timezone.utc).isoformat()
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                error = record['error']; stopping = True
        elif code == 75:
            record['status'] = 'paused'; stopping = True
        else:
            error = f'{run} exited {code}; inspect its training log'
            record.update(status='failed', error=error); stopping = True

    try:
        while True:
            stopping = stopping or is_paused()
            if stopping:
                signal_pause()
            for seed, item in list(active.items()):
                run, process, start = item
                code = process.poll()
                if code is not None:
                    del active[seed]
                    drain_record(run, code, time.perf_counter() - start)
                    save(active)
            if not stopping:
                for seed, runs in lanes.items():
                    if seed in active:
                        continue
                    run = next((r for r in runs if jobs.get('train/' + r, {}).get('status') != 'completed'), None)
                    if run is None:
                        continue
                    if is_paused():
                        stopping = True
                        break
                    record = jobs.setdefault('train/' + run, {})
                    process = launch(run)
                    record.update(status='running', pid=process.pid)
                    active[seed] = (run, process, time.perf_counter())
                    save(active)
            if not active:
                if error:
                    raise RuntimeError(error)
                return 75 if stopping else 0
            sleep(.5)
    except BaseException:
        # No orphan training process if launch/verification/coordinator fails.
        signal_pause()
        for run, process, start in active.values():
            code = process.wait()
            drain_record(run, code, time.perf_counter() - start)
        active.clear()
        save(active)
        raise


def status():
    q = read(DIRECTORY / 'queue.json')
    print('Queue:', q.get('state', 'not_started'), '| Mode:',q.get('execution_mode','serial_delegated_runs' if EXECUTOR=='server5090' else 'two_experiment_lanes'),'; final evaluations serial')
    print('Pause requested:', (DIRECTORY / 'PAUSE.request').exists())
    if ASSIGNMENT:print('Physical executor:',EXECUTOR,'| Local assigned runs:',RUNS,'| Delegated:',[r for r,o in ASSIGNMENT['owners'].items() if o!=EXECUTOR])
    jobs = q.get('jobs', {})
    print('Completed training:', sum(j.get('status') == 'completed' for k, j in jobs.items() if k.startswith('train/')), '/',len(RUNS))
    print('Completed evaluations:', sum(j.get('status') == 'completed' for k, j in jobs.items() if k.startswith('val/')), '/',len(RUNS))
    for run in RUNS:
        record = jobs.get('train/' + run, {})
        p = read(DIRECTORY / 'workers' / run / 'trainer_status.json')
        print(run, '|', record.get('status', 'queued'), '| epoch', p.get('epoch', '-'), '/100 | batch',
              p.get('step', '-'), '/', p.get('total_steps', '-'), '| phase', p.get('phase', '-'),
              '| loss', p.get('latest_loss', '-'), '| updated', p.get('updated_at', '-'))
    if q.get('execution_phase') == 'serial_evaluation':
        print('Evaluation:', q.get('current_job'))
        print(json.dumps(read(DIRECTORY / 'trainer_status.json'), ensure_ascii=False, indent=2))
    if q.get('error'):
        print('ERROR:', q['error'])
    print('Output: MedCLIPSeg/outputs_research/formal_detail_mix_kvasir_v1')


def work(action):
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from local_runtime import configure_local_models
    configure_local_models()
    from local_reproduce import acquire_gpu_queue
    from research_detail_formal.common import (assigned_runs,load_suite, suite_output, make_config, checkpoint_path,
                                            lock_plan, write_json, now, sha256)
    from research_detail_formal.runner import require_check, verify_checkpoint, report
    from train_detail_formal import TrainingPaused
    suite = load_suite()
    expected_runs=[r for r in assigned_runs(suite) if not ASSIGNMENT or ASSIGNMENT['owners'][r]==EXECUTOR]
    if sorted(RUNS)!=sorted(expected_runs) or len(RUNS)!=len(set(RUNS)):
        raise ValueError('Parallel lanes must cover each locked run exactly once')
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    path = DIRECTORY / 'queue.json'
    streams = []
    with acquire_gpu_queue():
        require_check(suite)
        plan = lock_plan(suite)
        q = read(path) or {'jobs': {}}
        if q.get('plan_digest') and q['plan_digest'] != plan['digest']:
            raise ValueError('Queue protocol mismatch')
        if action == 'train' and q.get('state') in ('running', 'paused', 'failed'):
            raise ValueError('Use Resume for existing work')
        if q.get('state') == 'completed':
            print('All jobs already completed.'); return 0
        # Checkpoint/config signatures exclude the operational control directory.
        # Only status/pause files move. RNG, optimizer, scheduler and weights resume unchanged.
        if action == 'resume':
            (DIRECTORY / 'PAUSE.request').unlink(missing_ok=True)
            for run in RUNS:
                (DIRECTORY / 'workers' / run / 'PAUSE.request').unlink(missing_ok=True)
        elif (DIRECTORY / 'PAUSE.request').exists():
            raise ValueError('Pause requested; use Resume')
        snapshot = suite_output(suite) / 'execution_history'
        snapshot.mkdir(exist_ok=True)
        stamp = str(time.time_ns())
        write_json(snapshot / f'{stamp}_queue_before.json', q)
        write_json(snapshot / f'{stamp}_parallel.json', dict(
            started_at=now(), mode='two_experiment_lanes', max_training_processes=2,
            experiment_lanes=LANES,
            final_evaluation='serial_after_all_training', scientific_plan_digest=plan['digest'],
            scheduler_sha256=sha256(Path(__file__)),
            launcher_sha256=sha256(WORKSPACE / 'tools/launch_research_detail_formal.py'),
            host=HOST,shared_protocol_digest=plan['shared_protocol']['digest'],
            benchmark=read(WORKSPACE/'control/research_detail_formal/formal_detail_mix_kvasir_v1'/HOST/'concurrency_check.json')))
        q.update(state='running', pid=os.getpid(), plan_digest=plan['digest'], error=None,
                 execution_mode='two_experiment_lanes', execution_phase='training')
        if ASSIGNMENT:q.update(physical_executor=EXECUTOR,execution_assignment=ASSIGNMENT,execution_mode='serial_delegated_runs' if EXECUTOR=='server5090' else 'two_experiment_lanes',assigned_runs=RUNS)
        q.pop('current_job', None)

        def save(active):
            q['active_jobs'] = {str(seed): {'run': run, 'pid': proc.pid} for seed, (run, proc, _) in active.items()}
            q['updated_at'] = now()
            write_json(path, q)

        def launch(run):
            cfg = make_config(suite, run)
            child_dir = DIRECTORY / 'workers' / run
            child_dir.mkdir(parents=True, exist_ok=True)
            args = [sys.executable, '-u', 'train_detail_formal.py', '--config-file',
                    str(suite_output(suite) / 'configs' / f'{run}.yaml'), '--output-dir', cfg.output_dir,
                    '--seed', str(cfg.seed), '--data_percentage', '100', '--num-workers', '0',
                    '--control-dir', str(child_dir), '--gradient-checkpointing']
            if checkpoint_path(suite, run).exists():
                args.append('--resume')
            stream = (suite_output(suite) / 'logs' / f'{run}.log').open('a', encoding='utf-8')
            streams.append(stream)
            startup = None
            if os.name=='nt':
                startup = subprocess.STARTUPINFO()
                startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup.wShowWindow = subprocess.SW_HIDE
            child = subprocess.Popen(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, startupinfo=startup)
            q['jobs'].setdefault('train/' + run, {}).setdefault('started_at', now())
            print('START', run, 'PID', child.pid, 'resume=', '--resume' in args, flush=True)
            return child

        try:
            save({})
            lanes = LANES
            code = schedule(lanes, q['jobs'], launch, lambda run: verify_checkpoint(suite, run, plan),
                            save, lambda: (DIRECTORY / 'PAUSE.request').exists(), pause_all)
            if code == 75:
                q['state'] = 'paused'; save({}); return code
            q['execution_phase'] = 'serial_evaluation'; save({})
            # Keep the same GPU lock throughout the handoff. MC evaluation is
            # unchanged and only runs after both training processes exit.
            from research_detail_formal.evaluate import evaluate
            for run in RUNS:
                digest = verify_checkpoint(suite, run, plan)
                if q['jobs']['train/' + run]['checkpoint_sha256'] != digest:
                    raise ValueError('Completed checkpoint changed: ' + run)
                for condition in suite['evaluation_conditions']:
                    if (DIRECTORY / 'PAUSE.request').exists():
                        raise TrainingPaused()
                    key = f'val/{run}/{condition}'
                    record = q['jobs'].setdefault(key, {})
                    if record.get('status') == 'completed':
                        continue
                    q['current_job'] = key
                    record.update(status='running', started_at=record.get('started_at', now()))
                    save({}); start = time.perf_counter()
                    try:
                        evaluate(make_config(suite, run), checkpoint_path(suite, run),
                                 suite_output(suite) / 'evaluations' / run / condition, DIRECTORY, condition)
                    finally:
                        record['active_seconds'] = record.get('active_seconds', 0.) + time.perf_counter() - start
                        save({})
                    record.update(status='completed', completed_at=now()); save({})
                    report(suite)
            print('Report:', report(suite), flush=True)
            q.update(state='completed', completed_at=now()); save({})
            return 0
        except TrainingPaused:
            key = q.get('current_job')
            if key and q['jobs'][key].get('status') != 'completed':
                q['jobs'][key]['status'] = 'paused'
            q['state'] = 'paused'; save({}); return 75
        except BaseException as exc:
            q.update(state='failed', error=f'{type(exc).__name__}: {exc}'); save({}); raise
        finally:
            for stream in streams:
                stream.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['train', 'resume', 'pause', 'status'], default='status')
    args = parser.parse_args()
    if args.action == 'pause':
        pause_all(); print('Pause requested for BOTH active experiments; wait for Queue: paused before powering off.'); return 0
    if args.action == 'status':
        status(); return 0
    return work(args.action)


if __name__ == '__main__':
    raise SystemExit(main())
