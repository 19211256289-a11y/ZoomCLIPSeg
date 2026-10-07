"""Evaluation-only probe: timing, ultrasound image scores, polyp second-pass variants, report. Shares the GPU lock."""
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
    """Checkpoint sources, reusing the loaders and hash checks of the accepted evaluations."""
    from research_zoom_ext.common import Source
    polyp, ultra, timing = protocol['polyp'], protocol['ultrasound'], protocol['timing']
    pool = {}
    for name in polyp['sources']:
        runs = [r for r in polyp['runs'] if polyp_source(protocol, r) == name]
        if name == timing['source']:
            runs = runs+[r for r in timing['runs'] if r not in runs]
        pool[name] = Source(dict(source=name, runs=runs))
    if timing['source'] not in pool:
        raise ValueError('The timing source must be one of the polyp sources')
    pool[ultra['source']] = Source(dict(source=ultra['source'], runs=ultra['runs']))
    return pool


def lock_plan(protocol, pool):
    """Seals protocol, code, checkpoints and cohorts. Computes no performance number."""
    from research_zoom_ext.common import seal_cohorts
    checkpoints, cohorts = {}, {}
    for name, source in pool.items():
        if not source.ready():
            raise ValueError(f'Checkpoints or accepted results of {name} are not on this PC')
        checkpoints[name] = source.expected_checkpoints()
        cohorts[name] = {t: {k: v for k, v in c.items() if k != 'groups'} for t, c in seal_cohorts(source).items()}
    ultra = protocol['ultrasound']
    val_list = ROOT/'data'/pool[ultra['source']].targets[0]/'Prompts_Folder'/ultra['val_list']
    plan = dict(protocol={k: v for k, v in protocol.items()}, protocol_sha256=sha256(PROTOCOL), code=code_manifest(),
                checkpoints=checkpoints, cohorts=cohorts, val_list_sha256=sha256(val_list),
                prompts={name: source.prompt for name, source in pool.items()})
    plan['digest'] = object_hash(plan)
    path = OUTPUT/'evaluation_plan.json'
    if path.exists() and read(path)['digest'] != plan['digest']:
        finished = [p for p in (OUTPUT/'evaluations').rglob('manifest.json') if maybe(p).get('complete')] if (OUTPUT/'evaluations').exists() else []
        if finished:
            raise ValueError('Protocol, code or inputs changed after results of this suite were produced; use a new suite name')
    write_json(path, plan)
    return plan


def load_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def reproduction(mine, old, tolerance, what):
    """mine, old: name -> DSC in [0, 1]. Raises if the mean differs by more than the tolerance."""
    if sorted(mine) != sorted(old):
        raise RuntimeError(f'{what}: case list differs from the accepted evaluation')
    difference = 100*(sum(mine.values())-sum(old.values()))/len(old)
    record = dict(what=what, DSC_percent=100*sum(mine.values())/len(mine), accepted_DSC_percent=100*sum(old.values())/len(old),
                  difference_pp=difference, largest_case_difference_pp=100*max(abs(mine[k]-old[k]) for k in old),
                  tolerance_pp=tolerance, passed=abs(difference) <= tolerance)
    if not record['passed']:
        raise RuntimeError(f'{what}: DSC differs from the accepted value by {difference:+.4f} pp')
    return record


def status():
    protocol = load_protocol() if PROTOCOL.exists() else None
    queue, progress = maybe(CONTROL/'queue.json'), maybe(CONTROL/'trainer_status.json')
    jobs, state = queue.get('jobs', {}), queue.get('state', 'not_started')
    print(f'{NAME}: {state} | pause requested: {(CONTROL/"PAUSE.request").exists()}')
    if state == 'waiting_for_gpu':
        print('    waiting: another queue holds the GPU (pause it; this evaluation then starts by itself)')
    if protocol:
        keys = job_keys(protocol)
        for part, title in (('timing/', 'timing'), ('detect/', 'ultrasound scores'), ('zoom/', 'polyp variants')):
            mine = [k for k in keys if k.startswith(part)]
            print(f"    {title}: {sum(jobs.get(k, {}).get('status') == 'completed' for k in mine)}/{len(mine)}")
    if state == 'running':
        line = f"    current: {queue.get('current_job')}"
        if progress.get('status') == 'running' and progress.get('total_steps'):
            line += f" [{progress.get('config')}] step {progress.get('step')}/{progress.get('total_steps')}"
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

        def pending(keys):
            return any(queue['jobs'].get(k, {}).get('status') != 'completed' for k in keys)

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
            from . import evaluate as probe
            polyp, ultra, clock = protocol['polyp'], protocol['ultrasound'], protocol['timing']
            tolerance, batch_size, zoom = protocol['reproduction_tolerance_pp'], protocol['batch_size'], protocol['zoom']
            reference = polyp['all_variants'][0]

            def use(source):
                """The accepted evaluation code reads the class sentence from these module constants."""
                zoom_evaluate.GENERIC = test_evaluate.GENERIC = source.prompt

            def test_cohort(source, target):
                folder = ROOT/'data'/target/'Test_Folder'
                records = read_text(ROOT/'data'/target/'Prompts_Folder'/source.test_list)
                if len(records) != plan['cohorts'][source.kind][target]['n']:
                    raise ValueError('Unexpected cohort size: '+target)
                return folder, records

            def with_model(source, run, keys, body):
                """Loads the checkpoint once for a group of jobs and always releases it."""
                if not pending(keys):
                    return
                if pause.exists():
                    raise TrainingPaused()
                use(source)
                queue['current_job'] = f'{run}/load'
                save()
                model, cfg, digest = source.load_model(run, plan['checkpoints'][source.kind][run])
                tracker = TrainingControl(str(CONTROL), run_id=run, phase='evaluation')
                try:
                    body(model, cfg, digest, tracker)
                    tracker.update('completed')
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

            def signature(**fields):
                return dict(suite=NAME, plan_digest=plan['digest'], **fields)

            # ---------------- file-only check
            def check():
                source = pool[clock['source']]
                use(source)
                folder, records = test_cohort(source, clock['target'])
                zoom_evaluate.check_preprocessing(folder, records)
                write_json(CONTROL/'check.json', dict(passed=True, checked_at=now()))
            job('check', check)

            # ---------------- Part C: timing
            source = pool[clock['source']]
            for run in clock['runs']:
                def body(model, cfg, digest, tracker, run=run, source=source):
                    folder, records = test_cohort(source, clock['target'])
                    def measure():
                        result = probe.timing(model, cfg, folder, records, reference, zoom, source.prompt,
                                              clock['warmup_images'], clock['measured_images'], tracker, 'timing')
                        (OUTPUT/'timing').mkdir(parents=True, exist_ok=True)
                        write_json(OUTPUT/'timing'/f'{run}.json', dict(result, run=run, checkpoint_sha256=digest,
                                                                       plan_digest=plan['digest']))
                    job(f'timing/{run}', measure)
                with_model(source, run, [f'timing/{run}'], body)

            # ---------------- Part B: ultrasound image scores
            source = pool[ultra['source']]
            dataset_name = source.targets[0]
            if ultra['cohorts'][1:] != source.targets:
                raise ValueError('The ultrasound cohorts differ from the test sets of the source suite')
            for run in ultra['runs']:
                def body(model, cfg, digest, tracker, run=run, source=source):
                    for cohort in ultra['cohorts']:
                        def score(cohort=cohort):
                            if cohort == 'Val':
                                folder = ROOT/'data'/dataset_name/'Val_Folder'
                                records = read_text(ROOT/'data'/dataset_name/'Prompts_Folder'/ultra['val_list'])
                            else:
                                folder, records = test_cohort(source, cohort)
                            zoom_evaluate.check_preprocessing(folder, records)
                            destination = OUTPUT/'evaluations'/'detect'/run/cohort
                            probe.image_scores(model, cfg, folder, records, source.prompt, batch_size, destination,
                                               signature(part='detect', run=run, cohort=cohort, checkpoint_sha256=digest,
                                                         prompt=source.prompt), tracker, cohort)
                            if cohort != 'Val':
                                mine = {r['name']: float(r['DSC']) for r in load_csv(destination/'scores.csv')}
                                old = {r['name']: r['DSC'] for r in load_cases(source.reference(run, cohort)/'cases.csv')}
                                write_json(destination/'reproduction.json',
                                           reproduction(mine, old, tolerance, f'{run}/{cohort} single pass'))
                        job(f'detect/{run}/{cohort}', score)
                with_model(source, run, [f'detect/{run}/{c}' for c in ultra['cohorts']], body)

            # ---------------- Part A: polyp second-pass variants
            for run in polyp['runs']:
                source = pool[polyp_source(protocol, run)]
                variants = variants_for(protocol, run)
                two_pass = ROOT/'outputs_research'/polyp['accepted_two_pass'][source.kind]/'evaluations'/run

                def body(model, cfg, digest, tracker, run=run, source=source, variants=variants, two_pass=two_pass):
                    for target in source.targets:
                        def evaluate(target=target):
                            folder, records = test_cohort(source, target)
                            zoom_evaluate.check_preprocessing(folder, records)
                            destination = OUTPUT/'evaluations'/'zoom'/run/target
                            probe.evaluate_variants(model, cfg, folder, records, variants, zoom, source.prompt, batch_size,
                                                    destination, signature(part='zoom', run=run, cohort=target,
                                                                           checkpoint_sha256=digest, prompt=source.prompt,
                                                                           variants=[v['name'] for v in variants]),
                                                    tracker, target)
                            rows = load_csv(destination/'cases.csv')
                            checks = []
                            mine = {r['name']: float(r['DSC']) for r in rows if r['config'] == 'single'}
                            old = {r['name']: r['DSC'] for r in load_cases(source.reference(run, target)/'cases.csv')}
                            checks.append(reproduction(mine, old, tolerance, f'{run}/{target} single pass'))
                            mine = {r['name']: float(r['DSC']) for r in rows if r['config'] == 't15'}
                            old = {r['name']: float(r['DSC']) for r in load_csv(two_pass/target/'cases.csv') if r['config'] == 't15'}
                            checks.append(reproduction(mine, old, tolerance, f'{run}/{target} accepted two-pass (t15)'))
                            write_json(destination/'reproduction.json', dict(checks=checks, passed=True))
                        job(f'zoom/{run}/{target}', evaluate)
                with_model(source, run, [f'zoom/{run}/{t}' for t in source.targets], body)

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
