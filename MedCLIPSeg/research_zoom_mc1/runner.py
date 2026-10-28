"""Evaluation-only analysis of SCZ with one stochastic pass per run (accuracy and time). Shares the GPU lock with every other queue."""
from local_runtime import configure_local_models
configure_local_models()
import argparse
import csv
import gc
import os
import time
import traceback

from .common import *


def sources(protocol):
    from research_zoom_ext.common import Source
    return {name: Source(dict(source=name, runs=[r for r in protocol['runs'] if source_of(protocol, r) == name]))
            for name in protocol['sources']}


def load_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def lock_plan(protocol, pool):
    """Seals protocol, code, checkpoints, cohorts and the accepted two-pass results. Computes no performance number."""
    from research_zoom_ext.common import seal_cohorts
    checkpoints, cohorts, accepted = {}, {}, {}
    for name, source in pool.items():
        if not source.ready():
            raise ValueError(f'Checkpoints or accepted results of {name} are not on this PC')
        checkpoints[name] = source.expected_checkpoints()
        cohorts[name] = {t: {k: v for k, v in c.items() if k != 'groups'} for t, c in seal_cohorts(source).items()}
        if list(source.targets) != list(TARGETS):
            raise ValueError('Unexpected test sets')
    for run in protocol['runs']:
        for target in TARGETS:
            path = accepted_two_pass(protocol, run, target)
            if not path.is_file():
                raise ValueError(f'Accepted two-pass result missing: {path}')
            accepted[f'{run}/{target}'] = sha256(path)
    plan = dict(protocol=dict(protocol), protocol_sha256=sha256(PROTOCOL), code=code_manifest(), checkpoints=checkpoints,
                cohorts=cohorts, accepted_two_pass=accepted, prompts={name: source.prompt for name, source in pool.items()})
    plan['digest'] = object_hash(plan)
    path = OUTPUT/'evaluation_plan.json'
    if path.exists() and read(path)['digest'] != plan['digest']:
        finished = [p for p in (OUTPUT/'evaluations').rglob('manifest.json') if maybe(p).get('complete')] if (OUTPUT/'evaluations').exists() else []
        if finished:
            raise ValueError('Protocol, code or inputs changed after results of this suite were produced; use a new suite name')
    write_json(path, plan)
    return plan


def status():
    queue, progress = maybe(CONTROL/'queue.json'), maybe(CONTROL/'trainer_status.json')
    jobs, state = queue.get('jobs', {}), queue.get('state', 'not_started')
    print(f'{NAME}: {state} | pause requested: {(CONTROL/"PAUSE.request").exists()}')
    if state == 'waiting_for_gpu':
        print('    waiting: another queue holds the GPU; this evaluation starts by itself when the GPU is free')
    if queue.get('n_jobs'):
        print(f"    jobs completed: {sum(j.get('status') == 'completed' for j in jobs.values())}/{queue['n_jobs']}")
    if state == 'running':
        line = f"    current: {queue.get('current_job')}"
        if progress.get('status') == 'running' and progress.get('config'):
            line += f" [{progress.get('config')}] batch {progress.get('step')}"
        print(line)
    if queue.get('error'):
        print('    ERROR:', queue['error'])
    if state == 'completed':
        print('    report:', REPORT/'report.md')
    print('    updated:', queue.get('updated_at', 'n/a'), '(UTC)')
    return state


def work():
    import torch
    from local_reproduce import acquire_gpu_queue
    from train_detail_formal import TrainingControl, TrainingPaused
    from utils.main_utils import read_text
    protocol = load_protocol()
    CONTROL.mkdir(parents=True, exist_ok=True)
    pause, path = CONTROL/'PAUSE.request', CONTROL/'queue.json'
    pause.unlink(missing_ok=True)
    queue = maybe(path) or dict(jobs={})
    if queue.get('state') == 'completed':
        print(NAME, 'already completed. Report:', REPORT/'report.md', flush=True)
        return 0
    queue.update(n_jobs=len(job_keys(protocol)), error=None, pid=os.getpid())

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
            print('Verifying checkpoints and test files, sealing the protocol ...', flush=True)
            pool = sources(protocol)
            plan = lock_plan(protocol, pool)
            if queue.get('plan_digest') not in (None, plan['digest']):
                queue['jobs'] = {}   # only reachable before any completed evaluation (see lock_plan)
            queue['plan_digest'] = plan['digest']
            save()
            import research_zoom.evaluate as zoom_evaluate
            import research_detail_test.evaluate as test_evaluate
            from research_detail_test.stats import load_cases
            from . import evaluate as comparison
            tolerance = protocol['reproduction_tolerance_pp']

            def cohort(source, target):
                folder = ROOT/'data'/target/'Test_Folder'
                records = read_text(ROOT/'data'/target/'Prompts_Folder'/source.test_list)
                if len(records) != plan['cohorts'][source.kind][target]['n']:
                    raise ValueError('Unexpected cohort size: '+target)
                return folder, records

            def check():
                source = pool[source_of(protocol, protocol['runs'][0])]
                zoom_evaluate.GENERIC = test_evaluate.GENERIC = source.prompt
                zoom_evaluate.check_preprocessing(*cohort(source, TARGETS[0]))
                write_json(CONTROL/'check.json', dict(passed=True, checked_at=now()))
            job('check', check)

            for run in protocol['runs']:
                keys = [f'eval/{run}/{target}' for target in TARGETS]
                if all(queue['jobs'].get(k, {}).get('status') == 'completed' for k in keys):
                    continue
                if pause.exists():
                    raise TrainingPaused()
                source = pool[source_of(protocol, run)]
                zoom_evaluate.GENERIC = test_evaluate.GENERIC = source.prompt
                queue['current_job'] = f'{run}/load'
                save()
                model, cfg, digest = source.load_model(run, plan['checkpoints'][source.kind][run])
                tracker = TrainingControl(str(CONTROL), run_id=run, phase='evaluation')

                def evaluate(target):
                    folder, records = cohort(source, target)
                    zoom_evaluate.check_preprocessing(folder, records)
                    destination = OUTPUT/'evaluations'/run/target
                    comparison.evaluate(model, cfg, folder, records, source.prompt, protocol, destination,
                                      dict(suite=NAME, plan_digest=plan['digest'], run=run, cohort=target,
                                           checkpoint_sha256=digest, prompt=source.prompt), tracker, target)
                    mine = {r['name']: float(r['DSC']) for r in load_csv(destination/'cases.csv') if r['config'] == REPRODUCTION_CONFIG}
                    old = {r['name']: r['DSC'] for r in load_cases(source.reference(run, target)/'cases.csv')}
                    if sorted(mine) != sorted(old):
                        raise RuntimeError(f'{run}/{target}: case list differs from the accepted evaluation')
                    difference = 100*(sum(mine.values())-sum(old.values()))/len(old)
                    write_json(destination/'reproduction.json', dict(
                        run=run, cohort=target, single_pass_DSC_percent=100*sum(mine.values())/len(mine),
                        accepted_DSC_percent=100*sum(old.values())/len(old), difference_pp=difference,
                        tolerance_pp=tolerance, passed=abs(difference) <= tolerance))
                    if abs(difference) > tolerance:
                        raise RuntimeError(f'{run}/{target}: single-pass DSC differs from the accepted value by {difference:+.4f} pp')

                try:
                    for target in TARGETS:
                        job(f'eval/{run}/{target}', lambda t=target: evaluate(t))
                    tracker.update('completed')
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
            def timing():
                spec = protocol['timing']
                run = spec['run']
                source = pool[source_of(protocol, run)]
                zoom_evaluate.GENERIC = test_evaluate.GENERIC = source.prompt
                folder, records = cohort(source, spec['dataset'])
                model, cfg, digest = source.load_model(run, plan['checkpoints'][source.kind][run])
                try:
                    tracker = TrainingControl(str(CONTROL), run_id=run, phase='timing')
                    comparison.timing(model, cfg, folder, records, source.prompt, protocol, OUTPUT/'timing.json', tracker)
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
            job('timing', timing)
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
            print('PAUSED. Start again with the same command; completed jobs are kept.', flush=True)
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
