"""One-time ETIS-Larib evaluation: ten accepted checkpoints, single-pass and locked two-pass inference."""
from local_runtime import configure_local_models
configure_local_models()
import argparse
import gc
import os
import time
import traceback

from .common import *


def sources(protocol):
    """Checkpoint sources, reusing the loaders and hash checks of the accepted evaluations."""
    from research_zoom_ext.common import Source
    result = {}
    for name in protocol['sources']:
        runs = [r for r in protocol['runs'] if source_of(protocol, r) == name]
        result[name] = Source(dict(source=name, runs=runs))
    return result


def lock_plan(protocol, pool):
    """Seals protocol, code, checkpoints and the cohort. Computes no performance number."""
    folder, records, seal = prepare_cohort(protocol)
    checkpoints = {}
    for name, source in pool.items():
        if not source.ready():
            raise ValueError(f'Checkpoints or accepted results of {name} are not on this PC')
        if source.prompt != protocol['prompt'] or source.small_threshold() != protocol['small_target_threshold']:
            raise ValueError(f'{name}: prompt or small-target threshold differs from the protocol')
        checkpoints.update(source.expected_checkpoints())
    if sorted(checkpoints) != sorted(protocol['runs']):
        raise ValueError('Unexpected set of checkpoints')
    plan = dict(protocol={k: v for k, v in protocol.items() if k not in ('runs', 'seeds')}, protocol_sha256=sha256(PROTOCOL),
                code=code_manifest(), runs=protocol['runs'], checkpoints=checkpoints, cohort=seal,
                overlap_with_known_data=overlap_with_known_data(protocol))
    plan['digest'] = object_hash(plan)
    path = OUTPUT/'evaluation_plan.json'
    if path.exists() and read(path)['digest'] != plan['digest']:
        if any(maybe(OUTPUT/'evaluations'/run/'ETIS/manifest.json').get('complete') for run in protocol['runs']):
            raise ValueError('Protocol, code or inputs changed after results on this dataset were produced. This evaluation is run once.')
    write_json(path, plan)
    return plan, folder, records


def check(folder, records):
    """Decoding and layout check on files only; no model is involved."""
    import cv2
    import numpy as np
    for record in (records[0], records[-1]):
        raw = cv2.imread(str(folder/'img'/record['Image']))
        mask = cv2.imread(str(folder/'label'/record['Ground Truth']), 0)
        if raw is None or mask is None or raw.ndim != 3 or raw.shape[:2] != mask.shape:
            raise RuntimeError('Image or mask cannot be read as expected: '+record['Image'])
        values = np.unique(mask)
        if not set(values.tolist()) <= {0, 255} and ((mask > 20) & (mask < 235)).mean() > .02:
            raise RuntimeError('The mask is not binary: '+record['Ground Truth'])
    return dict(passed=True, example_shape=list(raw.shape))


def status():
    queue, progress = maybe(CONTROL/'queue.json'), maybe(CONTROL/'trainer_status.json')
    jobs, state = queue.get('jobs', {}), queue.get('state', 'not_started')
    print(f'{NAME}: {state} | pause requested: {(CONTROL/"PAUSE.request").exists()}')
    if state == 'waiting_for_gpu':
        print('    waiting: another queue holds the GPU (pause it, this evaluation then starts by itself)')
    if queue.get('n_jobs'):
        print(f"    jobs completed: {sum(j.get('status') == 'completed' for j in jobs.values())}/{queue['n_jobs']}")
    if state == 'running':
        line = f"    current: {queue.get('current_job')}"
        if progress.get('status') == 'running' and progress.get('total_steps'):
            line += f", step {progress.get('step')}/{progress.get('total_steps')}"
        print(line)
    if queue.get('error'):
        print('    ERROR:', queue['error'])
    if state == 'completed':
        print('    verdict:', maybe(REPORT/'results.json').get('verdict'), '| report:', REPORT/'report.md')
    print('    updated:', queue.get('updated_at', 'n/a'), '(UTC)')
    return state


def work():
    import torch
    from local_reproduce import acquire_gpu_queue
    from train_detail_formal import TrainingControl, TrainingPaused
    protocol = load_protocol()
    CONTROL.mkdir(parents=True, exist_ok=True)
    pause, path = CONTROL/'PAUSE.request', CONTROL/'queue.json'
    pause.unlink(missing_ok=True)
    queue = maybe(path) or dict(jobs={})
    if queue.get('state') == 'completed':
        print(NAME, 'already completed. Report:', REPORT/'report.md', flush=True)
        return 0
    queue.update(n_jobs=len(protocol['runs'])+2, error=None, pid=os.getpid())

    def save():
        queue['updated_at'] = now()
        write_json(path, queue)

    while True:
        try:
            lock = acquire_gpu_queue()
            break
        except RuntimeError:
            queue['state'] = 'waiting_for_gpu'
            save()
            print('Waiting for the GPU (another queue is running) ...', flush=True)
            if pause.exists():
                queue['state'] = 'paused'
                save()
                return 75
            time.sleep(30)
    with lock:
        queue.update(state='running', started_at=queue.get('started_at', now()))

        def job(key, function):
            record = queue['jobs'].setdefault(key, {})
            if record.get('status') == 'completed':
                return
            if pause.exists():
                raise TrainingPaused()
            queue['current_job'] = key
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
            print('Verifying the dataset and sealing the protocol ...', flush=True)
            pool = sources(protocol)
            plan, folder, records = lock_plan(protocol, pool)
            if queue.get('plan_digest') not in (None, plan['digest']):
                queue['jobs'] = {}   # only reachable before any completed evaluation (see lock_plan)
            queue['plan_digest'] = plan['digest']
            save()
            job('check', lambda: write_json(CONTROL/'check.json', check(folder, records)))
            import research_zoom.evaluate as zoom_evaluate
            import research_detail_test.evaluate as test_evaluate
            if zoom_evaluate.GENERIC != protocol['prompt'] or test_evaluate.GENERIC != protocol['prompt']:
                raise RuntimeError('The class sentence of the evaluation code differs from the protocol')
            method = dict(configs=protocol['configs'], zoom=protocol['zoom'])
            zoom_evaluate.check_preprocessing(folder, records)
            for run in protocol['runs']:
                if queue['jobs'].get(run, {}).get('status') == 'completed':
                    continue
                if pause.exists():
                    raise TrainingPaused()
                queue['current_job'] = f'{run}/load'
                save()
                model, cfg, digest = pool[source_of(protocol, run)].load_model(run, plan['checkpoints'][run])
                tracker = TrainingControl(str(CONTROL), run_id=run, phase='evaluation')
                try:
                    job(run, lambda: zoom_evaluate.evaluate_condition(
                        model, cfg, folder, records, method, OUTPUT/'evaluations'/run/'ETIS',
                        dict(suite=NAME, run=run, cohort='ETIS', split='Test', checkpoint_sha256=digest,
                             plan_digest=plan['digest'], prompt=protocol['prompt']), tracker, 'ETIS'))
                    tracker.update('completed')
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
            from .report import write_report
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
            print('PAUSED. Start again with the same command.', flush=True)
            return 75
        except BaseException as exc:
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
        print(write_report())
        return 0
    return work()


if __name__ == '__main__':
    raise SystemExit(main())
