"""One process runs the requested suites in order. Each waits for its inputs and for the shared GPU lock."""
from local_runtime import configure_local_models
configure_local_models()
import argparse
import gc
import os
import time
import traceback

from .common import *


def job_keys(suite, source):
    return [f'{run}/{target}' for run in suite['runs'] for target in source.targets]+['report']


def lock_plan(suite, source):
    """Seals inputs and resolves the configuration. Computes no performance number."""
    for run in suite['runs']:
        for target in source.targets:
            folder = source.reference(run, target)
            manifest = read(folder/'manifest.json')
            if not manifest.get('complete') or sha256(folder/'cases.csv') != manifest['cases_sha256']:
                raise ValueError(f'Accepted single-pass result missing or modified: {folder}')
    configs, calibration = resolve_configs(suite, source)
    cohorts = seal_cohorts(source)
    plan = dict(protocol={k: v for k, v in suite.items() if k != 'runs'}, protocol_sha256=sha256(suite_file(suite['name'])),
                code=code_manifest(), runs=suite['runs'], checkpoints=source.expected_checkpoints(), configs=configs,
                calibration=calibration, prompt=source.prompt, small_threshold=source.small_threshold(),
                cohorts={t: {k: v for k, v in c.items() if k != 'groups'} for t, c in cohorts.items()},
                source_plan_digest=source.plan()['digest'])
    plan['digest'] = object_hash(plan)
    path = output(suite['name'])/'evaluation_plan.json'
    if path.exists() and read(path)['digest'] != plan['digest']:
        if any(maybe(output(suite['name'])/'evaluations'/run/target/'manifest.json').get('complete')
               for run in suite['runs'] for target in source.targets):
            raise ValueError(f"{suite['name']}: protocol, code or inputs changed after Test results were produced; use a new suite name")
    write_json(path, plan)
    return plan, cohorts


def status(names):
    for name in names:
        queue, progress = maybe(control(name)/'queue.json'), maybe(control(name)/'trainer_status.json')
        jobs, state = queue.get('jobs', {}), queue.get('state', 'not_started')
        print(f'{name}: {state} | pause requested: {(control(name)/"PAUSE.request").exists()}')
        if state == 'waiting_for_training_queue':
            print('    waiting: the local training queue (BUSI) has not finished')
        elif state == 'waiting_for_source':
            print('    waiting: checkpoints or single-pass Test results of the source suite are not on this PC yet')
        elif state == 'waiting_for_gpu':
            print('    waiting: another queue holds the GPU')
        total = queue.get('n_jobs')
        if total:
            print(f"    jobs completed: {sum(j.get('status') == 'completed' for j in jobs.values())}/{total}")
        if state == 'running':
            line = f"    current: {queue.get('current_job')}"
            if progress.get('status') == 'running' and progress.get('total_steps'):
                line += f", step {progress.get('step')}/{progress.get('total_steps')}"
            print(line)
        if queue.get('error'):
            print('    ERROR:', queue['error'])
        if state == 'completed':
            result = maybe(report_dir(name)/'results.json')
            print('    verdict:', result.get('verdict'), '| report:', report_dir(name)/'report.md')
        print('    updated:', queue.get('updated_at', 'n/a'), '(UTC)')


def run_suite(name, now_flag):
    import torch
    from local_reproduce import acquire_gpu_queue
    from train_detail_formal import TrainingControl, TrainingPaused
    from utils.main_utils import read_text
    suite = load_suite(name)
    source = Source(suite)
    folder = control(name)
    folder.mkdir(parents=True, exist_ok=True)
    pause, path = folder/'PAUSE.request', folder/'queue.json'
    queue = maybe(path) or dict(jobs={})
    if queue.get('state') == 'completed':
        print(name, 'already completed', flush=True)
        return 0
    queue.update(n_jobs=len(job_keys(suite, source)), error=None, pid=os.getpid())

    def save():
        queue['updated_at'] = now()
        write_json(path, queue)

    def wait(state):
        queue['state'] = state
        save()
        if pause.exists():
            queue['state'] = 'paused'
            save()
            return True
        time.sleep(60)
        return False

    while True:
        if not now_flag and TRAINING_QUEUE.exists() and read(TRAINING_QUEUE).get('state') != 'completed':
            if wait('waiting_for_training_queue'):
                return 75
            continue
        if not source.ready():
            if wait('waiting_for_source'):
                return 75
            continue
        try:
            lock = acquire_gpu_queue()
            break
        except RuntimeError:
            if wait('waiting_for_gpu'):
                return 75
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
            print('COMPLETE', name, key, flush=True)

        save()
        try:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is not available')
            queue['current_job'] = 'lock'
            save()
            plan, cohorts = lock_plan(suite, source)
            if queue.get('plan_digest') not in (None, plan['digest']):
                queue['jobs'] = {}  # a fix before any completed Test evaluation: every job is redone
            queue['plan_digest'] = plan['digest']
            save()
            # The accepted evaluation code reads the class sentence from these module constants.
            import research_zoom.evaluate as zoom_evaluate
            import research_detail_test.evaluate as test_evaluate
            zoom_evaluate.GENERIC = test_evaluate.GENERIC = source.prompt
            from research_detail_test.stats import load_cases
            from .report import load, write_report
            method = dict(configs=plan['configs'], zoom=suite['zoom'])
            tolerance = suite['single_pass_reproduction_tolerance_pp']
            destination_root = output(name)/'evaluations'
            for run in suite['runs']:
                keys = [f'{run}/{target}' for target in source.targets]
                if all(queue['jobs'].get(k, {}).get('status') == 'completed' for k in keys):
                    continue
                if pause.exists():
                    raise TrainingPaused()
                queue['current_job'] = f'{run}/load'
                save()
                model, cfg, digest = source.load_model(run, plan['checkpoints'][run])
                tracker = TrainingControl(str(folder), run_id=run, phase='evaluation')

                def evaluate(target):
                    data = ROOT/'data'/target/'Test_Folder'
                    records = read_text(ROOT/'data'/target/'Prompts_Folder'/source.test_list)
                    if len(records) != cohorts[target]['n']:
                        raise ValueError('Unexpected cohort size: '+target)
                    zoom_evaluate.check_preprocessing(data, records)
                    destination = destination_root/run/target
                    zoom_evaluate.evaluate_condition(
                        model, cfg, data, records, method, destination,
                        dict(suite=name, run=run, cohort=target, split='Test', checkpoint_sha256=digest,
                             plan_digest=plan['digest'], prompt=source.prompt), tracker, target)
                    mine = {r['name']: r['DSC'] for r in load(destination/'cases.csv') if r['config'] == 'single'}
                    old = {r['name']: r['DSC'] for r in load_cases(source.reference(run, target)/'cases.csv')}
                    if sorted(mine) != sorted(old):
                        raise RuntimeError(f'{run}/{target}: case list differs from the accepted single-pass evaluation')
                    difference = 100*(sum(mine.values())-sum(old.values()))/len(old)
                    write_json(destination/'reproduction.json', dict(
                        run=run, cohort=target, single_pass_DSC_percent=100*sum(mine.values())/len(mine),
                        accepted_DSC_percent=100*sum(old.values())/len(old), difference_pp=difference,
                        largest_case_difference_pp=100*max(abs(mine[k]-old[k]) for k in old), tolerance_pp=tolerance,
                        passed=abs(difference) <= tolerance))
                    if abs(difference) > tolerance:
                        raise RuntimeError(f'{run}/{target}: single-pass DSC differs from the accepted value by {difference:+.4f} pp')

                try:
                    for target in source.targets:
                        job(f'{run}/{target}', lambda t=target: evaluate(t))
                    tracker.update('completed')
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
            job('report', lambda: write_report(name))
            queue.update(state='completed', completed_at=now(), current_job=None)
            save()
            print('Report:', report_dir(name)/'report.md', flush=True)
            return 0
        except TrainingPaused:
            queue['state'] = 'paused'
            current = queue.get('current_job')
            if current in queue['jobs'] and queue['jobs'][current].get('status') != 'completed':
                queue['jobs'][current]['status'] = 'paused'
            save()
            print('PAUSED', name, flush=True)
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
    parser.add_argument('--suites', default='all', help='comma-separated suite names, run in this order')
    parser.add_argument('--now', action='store_true', help='do not wait for the local training queue to finish')
    args = parser.parse_args()
    os.chdir(ROOT)
    names = list(SUITES) if args.suites == 'all' else args.suites.split(',')
    if any(name not in SUITES for name in names):
        raise SystemExit('Unknown suite. Choose from: '+', '.join(SUITES))
    if args.action == 'status':
        status(names)
        return 0
    if args.action == 'pause':
        for name in names:
            control(name).mkdir(parents=True, exist_ok=True)
            (control(name)/'PAUSE.request').write_text(now(), encoding='utf-8')
        print('Pause requested for:', ', '.join(names))
        print('Wait until no suite shows "running" before powering off.')
        return 0
    if args.action == 'report':
        from .report import write_report
        for name in names:
            print(write_report(name))
        return 0
    for name in names:   # starting again clears an earlier pause request
        (control(name)/'PAUSE.request').unlink(missing_ok=True)
    for name in names:
        code = run_suite(name, args.now)
        if code:
            return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
