"""Queue: lock inputs, 15 two-pass Test evaluations (3 checkpoints x 5 cohorts), report. Waits for the shared GPU lock."""
from local_runtime import configure_local_models
configure_local_models()
import argparse
import gc
import os
import time
import traceback
from pathlib import Path

from .common import *

JOBS = [f'{run}/{target}' for run in RUNS for target in TARGETS]+['report']


def lock_plan(protocol, accepted):
    """Seals checkpoints and input files against the accepted Test evaluation. Computes no performance number."""
    earlier = read(ACCEPTED_TEST/'evaluation_plan.json')
    cohorts = {}
    for target in TARGETS:
        cohort = earlier['test_cohorts'][target]
        text = ROOT/'data'/target/'Prompts_Folder'/protocol['test_list']
        if sha256(text) != cohort['prompt_sha256']:
            raise ValueError(f'{target}: test list differs from the accepted Test evaluation')
        for path, digest in cohort['files'].items():
            if sha256(path) != digest:
                raise ValueError(f'{target}: test file changed since the accepted Test evaluation: {path}')
        cohorts[target] = dict(n=cohort['n'], prompt_sha256=cohort['prompt_sha256'], files_digest=object_hash(cohort['files']))
        for run in RUNS:
            folder = ACCEPTED_TEST/'evaluations'/run/target
            manifest = read(folder/'manifest.json')
            if not manifest.get('complete') or sha256(folder/'cases.csv') != manifest['cases_sha256']:
                raise ValueError(f'Accepted single-pass Test result missing or modified: {folder}')
    plan = dict(protocol=protocol, protocol_sha256=sha256(PROTOCOL), code=code_manifest(), test_cohorts=cohorts,
                accepted_test_plan_digest=earlier['digest'],
                checkpoints={run: accepted[run]['checkpoint_sha256'] for run in RUNS})
    plan['digest'] = object_hash(plan)
    path = OUTPUT/'evaluation_plan.json'
    if path.exists() and read(path)['digest'] != plan['digest']:
        if any(maybe(OUTPUT/'evaluations'/run/target/'manifest.json').get('complete') for run in RUNS for target in TARGETS):
            raise ValueError('Protocol, code or inputs changed after Test results were produced; use a new suite name')
    write_json(path, plan)
    return plan


def status():
    queue, progress = maybe(CONTROL/'queue.json'), maybe(CONTROL/'trainer_status.json')
    jobs = queue.get('jobs', {})
    state = queue.get('state', 'not_started')
    print('Suite:', NAME)
    print('Queue:', state, '| Pause requested:', (CONTROL/'PAUSE.request').exists())
    if state == 'waiting_for_gpu':
        print('Waiting for the GPU: another queue is still running. This evaluation starts automatically when it finishes.')
    print(f"Jobs completed: {sum(jobs.get(k, {}).get('status') == 'completed' for k in JOBS)}/{len(JOBS)}")
    if state == 'running':
        line = f"Current: {queue.get('current_job')}"
        if progress.get('status') == 'running' and progress.get('total_steps'):
            line += f", step {progress.get('step')}/{progress.get('total_steps')}"
        print(line)
    for run in RUNS:
        checks = [maybe(OUTPUT/'evaluations'/run/target/'reproduction.json') for target in TARGETS]
        checks = [c for c in checks if c]
        if checks:
            print(f"  {run}: single-pass vs accepted, largest difference {max(abs(c['difference_pp']) for c in checks):.4f} pp over {len(checks)} cohorts")
    if queue.get('error'):
        print('ERROR:', queue['error'])
    if state == 'completed':
        print('Verdict:', maybe(REPORT/'results.json').get('verdict'), '| Report:', REPORT/'report.md')
    print('Updated:', queue.get('updated_at', 'n/a'), '(UTC)')
    return state


def work(action):
    import torch
    from local_reproduce import acquire_gpu_queue
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
    path = CONTROL/'queue.json'
    queue = maybe(path) or dict(jobs={})

    def save():
        queue['updated_at'] = now()
        write_json(path, queue)

    while True:  # the GPU lock is shared with every other queue in this workspace
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
            protocol = load_protocol()
            accepted = accepted_runs(load_test_protocol())
            queue['current_job'] = 'lock'
            save()
            plan = lock_plan(protocol, accepted)
            if queue.get('plan_digest') not in (None, plan['digest']):
                queue['jobs'] = {}  # a fix before any completed Test evaluation: every job is redone
            queue['plan_digest'] = plan['digest']
            save()
            from research_zoom.evaluate import check_preprocessing, evaluate_condition
            from research_detail_test.evaluate import load_model
            from research_detail_test.stats import load_cases
            from .report import load, write_report
            method = dict(configs=[protocol['config']], zoom=protocol['zoom'])
            tolerance = protocol['single_pass_reproduction_tolerance_pp']
            for run in RUNS:
                keys = [f'{run}/{target}' for target in TARGETS]
                if all(queue['jobs'].get(k, {}).get('status') == 'completed' for k in keys):
                    continue
                if pause.exists():
                    raise TrainingPaused()
                queue['current_job'] = f'{run}/load'
                save()
                model, cfg, digest = load_model(run, accepted[run]['checkpoint_sha256'])
                control = TrainingControl(str(CONTROL), run_id=run, phase='evaluation')

                def evaluate(target):
                    folder = ROOT/'data'/target/'Test_Folder'
                    records = read_text(ROOT/'data'/target/'Prompts_Folder'/protocol['test_list'])
                    if len(records) != plan['test_cohorts'][target]['n']:
                        raise ValueError('Unexpected cohort size: '+target)
                    check_preprocessing(folder, records)
                    destination = OUTPUT/'evaluations'/run/target
                    evaluate_condition(model, cfg, folder, records, method, destination,
                                       dict(run=run, cohort=target, split='Test', checkpoint_sha256=digest,
                                            plan_digest=plan['digest']), control, target)
                    mine = {r['name']: r['DSC'] for r in load(destination/'cases.csv') if r['config'] == 'single'}
                    old = {r['name']: r['DSC'] for r in load_cases(ACCEPTED_TEST/'evaluations'/run/target/'cases.csv')}
                    if sorted(mine) != sorted(old):
                        raise RuntimeError(f'{run}/{target}: case list differs from the accepted Test evaluation')
                    difference = 100*(sum(mine.values())-sum(old.values()))/len(old)
                    write_json(destination/'reproduction.json', dict(
                        run=run, cohort=target, single_pass_DSC_percent=100*sum(mine.values())/len(mine),
                        accepted_DSC_percent=100*sum(old.values())/len(old), difference_pp=difference,
                        largest_case_difference_pp=100*max(abs(mine[k]-old[k]) for k in old), tolerance_pp=tolerance,
                        passed=abs(difference) <= tolerance))
                    if abs(difference) > tolerance:
                        raise RuntimeError(f'{run}/{target}: single-pass DSC differs from the accepted Test value by {difference:+.4f} pp')

                for target in TARGETS:
                    job(f'{run}/{target}', lambda t=target: evaluate(t))
                control.update('completed')
                del model
                gc.collect()
                torch.cuda.empty_cache()
            job('report', write_report)
            queue.update(state='completed', completed_at=now(), current_job=None)
            save()
            print('Report:', REPORT/'report.md', flush=True)
            return 0
        except TrainingPaused:
            queue['state'] = 'paused'
            current = queue.get('current_job')
            if current in queue['jobs'] and queue['jobs'][current].get('status') != 'completed':
                queue['jobs'][current]['status'] = 'paused'
            save()
            print('PAUSED. Use ResearchZoomTest.cmd -Action Resume.', flush=True)
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
