"""Queue of pilot C: lock, build cohorts, preprocessing check, 15 evaluations, report. Waits for the shared GPU lock."""
from local_runtime import configure_local_models
configure_local_models()
import argparse
import gc
import os
import time
import traceback
from pathlib import Path

from .common import *


def jobs_expected(protocol):
    return ['cohorts', 'check']+[f"{run}/{c['name']}" for run in RUNS for c in protocol['conditions']]+['report']


def lock_plan(protocol, accepted, cfg):
    text = Path(cfg.DATASET.TEXT_PROMPT_PATH)/'Val_text.xlsx'
    plan = dict(protocol=protocol, protocol_sha256=sha256(PROTOCOL), code=code_manifest(), val_list_sha256=sha256(text),
                checkpoints={run: accepted[run]['checkpoint_sha256'] for run in RUNS})
    plan['digest'] = object_hash(plan)
    path = OUTPUT/'evaluation_plan.json'
    if path.exists() and read(path)['digest'] != plan['digest']:
        done = [maybe(OUTPUT/'evaluations'/run/c['name']/'manifest.json').get('complete') for run in RUNS for c in protocol['conditions']]
        if all(done):
            raise ValueError('Protocol or code changed after the pilot was completed; use a new suite name')
    write_json(path, plan)
    return plan


def status():
    protocol = load_protocol()
    queue, progress = maybe(CONTROL/'queue.json'), maybe(CONTROL/'trainer_status.json')
    jobs, expected = queue.get('jobs', {}), jobs_expected(protocol)
    state = queue.get('state', 'not_started')
    print('Suite:', NAME)
    print('Queue:', state, '| Pause requested:', (CONTROL/'PAUSE.request').exists())
    if state == 'waiting_for_gpu':
        print('Waiting for the GPU: another queue is still running. This pilot starts automatically when it finishes.')
    print(f"Jobs completed: {sum(jobs.get(k, {}).get('status') == 'completed' for k in expected)}/{len(expected)}")
    if state == 'running':
        line = f"Current: {queue.get('current_job')}"
        if progress.get('status') == 'running' and progress.get('total_steps'):
            line += f", step {progress.get('step')}/{progress.get('total_steps')}"
        print(line)
    for run in RUNS:
        check = maybe(OUTPUT/'evaluations'/run/'s100/val_check.json')
        if check:
            print(f"  Val check {run}: {check['local_DSC_percent']:.4f} vs accepted {check['accepted_DSC_percent']:.4f} ({check['difference_pp']:+.4f} pp)")
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
    from research_detail_formal.common import load_suite, make_config
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
            cfg0 = make_config(load_suite(), RUNS[0])
            plan = lock_plan(protocol, accepted, cfg0)
            if queue.get('plan_digest') not in (None, plan['digest']):
                queue['jobs'] = {}  # protocol or code changed before completion: every job is redone
            queue['plan_digest'] = plan['digest']
            save()
            from .evaluate import build_cohort, check_preprocessing, evaluate_condition
            from research_detail_test.evaluate import load_model
            records = read_text(Path(cfg0.DATASET.TEXT_PROMPT_PATH)/'Val_text.xlsx')
            if len(records) != 100:
                raise ValueError('Unexpected Val list')
            source = Path(cfg0.DATASET.VAL_PATH)
            cohorts = {}

            def build():
                for condition in protocol['conditions']:
                    cohorts[condition['name']] = build_cohort(condition, source, records, OUTPUT/'cohorts'/condition['name'])

            job('cohorts', build)
            if not cohorts:
                build()  # completed earlier: reload the saved cohort lists
            job('check', lambda: check_preprocessing(source, records))
            tolerance = protocol['val_reproduction_tolerance_pp']
            for run in RUNS:
                keys = [f"{run}/{c['name']}" for c in protocol['conditions']]
                if all(queue['jobs'].get(k, {}).get('status') == 'completed' for k in keys):
                    continue
                if pause.exists():
                    raise TrainingPaused()
                queue['current_job'] = f'{run}/load'
                save()
                model, cfg, digest = load_model(run, accepted[run]['checkpoint_sha256'])
                control = TrainingControl(str(CONTROL), run_id=run, phase='evaluation')

                def evaluate(name):
                    folder, cohort = cohorts[name]
                    destination = OUTPUT/'evaluations'/run/name
                    evaluate_condition(model, cfg, folder, cohort, protocol, destination,
                                       dict(run=run, condition=name, checkpoint_sha256=digest, plan_digest=plan['digest']),
                                       control, name)
                    if name == 's100':
                        from .report import load
                        rows = [r for r in load(destination/'cases.csv') if r['config'] == 'single']
                        local = 100*sum(r['DSC'] for r in rows)/len(rows)
                        expected = accepted[run]['summary']['DSC_percent']
                        write_json(destination/'val_check.json', dict(run=run, local_DSC_percent=local,
                                   accepted_DSC_percent=expected, difference_pp=local-expected, tolerance_pp=tolerance))
                        if abs(local-expected) > tolerance:
                            raise RuntimeError(f'{run}: single-pass Val DSC differs from the accepted value by {local-expected:+.4f} pp')

                # The unshifted cohort goes first: it validates loading and the pipeline before anything else.
                for name in ['s100']+[c['name'] for c in protocol['conditions'] if c['name'] != 's100']:
                    job(f'{run}/{name}', lambda n=name: evaluate(n))
                control.update('completed')
                del model
                gc.collect()
                torch.cuda.empty_cache()
            from .report import write_report
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
            print('PAUSED. Use ResearchZoom.cmd -Action Resume.', flush=True)
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
