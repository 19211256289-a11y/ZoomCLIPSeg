"""Queue of SCZ applied to the Polyp-PVT reference models (evaluation only). Shares the GPU lock with every other queue."""
import argparse
import gc
import os
import time
import traceback

import csv

from research_polyp_pvt import common as pvt
from .common import *


def lock_plan(protocol, source):
    """Seals protocol, code, the Polyp-PVT weights and their accepted single-pass results, and the test cohorts.
    Computes no performance number."""
    from research_zoom_ext.common import Source, seal_cohorts
    from research_zoom_etis import common as etis
    pvt_plan = read(pvt.OUTPUT/'plan.json')
    weights, accepted = {}, {}
    for seed in protocol['seeds']:
        manifest = maybe(pvt.training_dir(seed)/'manifest.json')
        if not manifest.get('complete') or manifest['plan_digest'] != pvt_plan['digest']:
            raise ValueError(f'Polyp-PVT seed {seed} is not trained yet; run this after polyp_pvt_kvasir_v1 has completed')
        weights[seed] = manifest['sha256']
        for target in ALL:
            folder = pvt.evaluation_dir(seed, target)
            if not maybe(folder/'manifest.json').get('complete'):
                raise ValueError(f'Accepted single-pass result of Polyp-PVT missing: seed {seed}, {target}')
            accepted[f'seed{seed}/{target}'] = sha256(folder/'cases.csv')
    cohort_source = Source(dict(source='kvasir_v1', runs=[]))
    if tuple(cohort_source.targets) != TARGETS:
        raise ValueError('Unexpected test sets')
    cohorts = {t: {k: v for k, v in c.items() if k != 'groups'} for t, c in seal_cohorts(cohort_source).items()}
    held_folder, held_records, held_seal = etis.prepare_cohort(etis.load_protocol())
    cohorts[HELD_OUT] = held_seal
    if cohorts != pvt_plan['cohorts']:
        raise ValueError('The test cohorts differ from those of the Polyp-PVT evaluation')
    plan = dict(protocol=protocol, protocol_sha256=sha256(PROTOCOL), code=code_manifest(), pvt_plan_digest=pvt_plan['digest'],
                weights={str(k): v for k, v in weights.items()}, accepted_single_pass=accepted, cohorts=cohorts,
                test_list=cohort_source.test_list)
    plan['digest'] = object_hash(plan)
    path = OUTPUT/'plan.json'
    if path.exists() and read(path)['digest'] != plan['digest']:
        finished = [p for p in OUTPUT.rglob('manifest.json') if maybe(p).get('complete')]
        if finished:
            raise ValueError('Protocol, code or inputs changed after results of this suite were produced; use a new suite name')
    write_json(path, plan)
    return plan, (held_folder, held_records)


def status():
    queue, progress = maybe(CONTROL/'queue.json'), maybe(CONTROL/'trainer_status.json')
    jobs, state = queue.get('jobs', {}), queue.get('state', 'not_started')
    print(f'{NAME}: {state} | pause requested: {(CONTROL/"PAUSE.request").exists()}')
    if state == 'waiting_for_gpu':
        print('    waiting: another queue holds the GPU; this queue starts by itself when the GPU is free')
    if queue.get('n_jobs'):
        print(f"    jobs completed: {sum(j.get('status') == 'completed' for j in jobs.values())}/{queue['n_jobs']}")
    if state == 'running':
        line = f"    current: {queue.get('current_job')}"
        if progress.get('status') == 'running' and progress.get('total_steps'):
            line += f" image {progress.get('step')}/{progress.get('total_steps')}"
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
    from utils.main_utils import read_text
    protocol, source_protocol = load_protocol()
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
                raise Paused()
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
            print('Verifying the Polyp-PVT weights and the test files; sealing the protocol ...', flush=True)
            plan, held = lock_plan(protocol, source_protocol)
            if queue.get('plan_digest') not in (None, plan['digest']):
                queue['jobs'] = {}   # only reachable before any completed evaluation (see lock_plan)
            queue['plan_digest'] = plan['digest']
            save()
            from research_polyp_pvt import evaluate as pvt_evaluation
            from . import evaluate as evaluation
            control = Control(CONTROL)

            def cohort(target):
                if target == HELD_OUT:
                    return held
                folder = ROOT/'data'/target/'Test_Folder'
                records = read_text(ROOT/'data'/target/'Prompts_Folder'/plan['test_list'])
                if len(records) != plan['cohorts'][target]['n']:
                    raise ValueError('Unexpected cohort size: '+target)
                return folder, records

            def check():
                from research_zoom.geometry import window_rule
                kind, side = window_rule(.01, (.4, .4, .5, .5), protocol['target_fraction'], protocol['zoom'], False)
                if kind != 'in' or abs(side-(.01/.15)**.5) > 1e-9:
                    raise RuntimeError('Unexpected window rule')
                write_json(CONTROL/'check.json', dict(passed=True, checked_at=now()))
            job('check', check)

            def load_csv(path):
                with open(path, newline='', encoding='utf-8-sig') as stream:
                    return list(csv.DictReader(stream))

            for seed in protocol['seeds']:
                keys = [f'zoom/seed{seed}/{t}' for t in ALL]
                if all(queue['jobs'].get(k, {}).get('status') == 'completed' for k in keys):
                    continue
                if pause.exists():
                    raise Paused()
                queue['current_job'] = f'zoom/seed{seed}/load'
                save()
                model, digest = pvt_evaluation.load_model(seed, plan['pvt_plan_digest'])
                if digest != plan['weights'][str(seed)]:
                    raise RuntimeError('Weights changed')

                def run(target, seed=seed, model=model, digest=digest):
                    folder, records = cohort(target)
                    destination = evaluation_dir(seed, target)
                    evaluation.evaluate(model, folder, records, protocol, destination,
                                        dict(suite=NAME, plan_digest=plan['digest'], seed=seed, cohort=target, weights_sha256=digest),
                                        control, target)
                    mine = {r['name']: float(r['DSC']) for r in load_csv(destination/'cases.csv') if r['config'] == 'single'}
                    old = {r['name']: float(r['DSC']) for r in load_csv(pvt.evaluation_dir(seed, target)/'cases.csv')}
                    if sorted(mine) != sorted(old):
                        raise RuntimeError(f'seed {seed}/{target}: case list differs from the accepted Polyp-PVT evaluation')
                    difference = 100*(sum(mine.values())-sum(old.values()))/len(old)
                    tolerance = protocol['reproduction_tolerance_pp']
                    write_json(destination/'reproduction.json', dict(
                        seed=seed, cohort=target, single_pass_DSC_percent=100*sum(mine.values())/len(mine),
                        accepted_DSC_percent=100*sum(old.values())/len(old), difference_pp=difference,
                        largest_case_difference_pp=100*max(abs(mine[k]-old[k]) for k in old),
                        tolerance_pp=tolerance, passed=abs(difference) <= tolerance))
                    if abs(difference) > tolerance:
                        raise RuntimeError(f'seed {seed}/{target}: single-pass DSC differs from the accepted value by {difference:+.4f} pp')

                try:
                    for target in ALL:
                        job(f'zoom/seed{seed}/{target}', lambda t=target: run(t))
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
            control.update('completed')
            from .report import write_report
            job('report', write_report)
            queue.update(state='completed', completed_at=now(), current_job=None)
            save()
            print('ALL COMPLETE. Report:', REPORT/'report.md', flush=True)
            return 0
        except Paused:
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
