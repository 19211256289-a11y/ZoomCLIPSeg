"""Queue of two further seeds of the Polyp-PVT reference: for every seed training and evaluation, then the report.
Shares the GPU lock with every other queue; can be paused and continued."""
import argparse
import gc
import os
import time
import traceback

from .common import *


def lock_plan(protocol):
    """Seals protocol, code, backbone weights, training lists, test cohorts and the accepted results that the report
    compares with. Computes no performance number."""
    from research_zoom_ext.common import Source, seal_cohorts
    from research_polyp_pvt.data import train_seal
    from research_zoom_etis import common as etis
    from .work import count_training_images
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f'Backbone weights not found: {WEIGHTS}')
    groups, sources = {}, {}
    for name, spec in protocol['groups'].items():
        source = sources.setdefault(spec['cohort_source'], Source(dict(source=spec['cohort_source'], runs=[])))
        if list(source.targets) != list(spec['targets']):
            raise ValueError(f'{name}: test sets differ from the accepted evaluation')
        train_n = count_training_images(protocol, name)
        cohorts = {t: {k: v for k, v in c.items() if k != 'groups'} for t, c in seal_cohorts(source).items()}
        groups[name] = dict(train_n=train_n, train=train_seal(group_protocol(protocol, name, train_n)), cohorts=cohorts,
                            test_list=source.test_list)
    held_folder, held_records, held_seal = etis.prepare_cohort(etis.load_protocol())
    for group in groups.values():
        group['cohorts'][protocol['held_out_target']] = held_seal
    accepted = {}
    for group, models in protocol['comparison'].items():
        for model in models:
            for seed in protocol['seeds']:
                for target in protocol['groups'][group]['targets']:
                    path, _ = accepted_cases(protocol, group, model, seed, target)
                    if not path.is_file():
                        raise ValueError(f'Accepted result missing: {path}')
                    accepted[f'{group}/{model}/seed{seed}/{target}'] = sha256(path)
    plan = dict(protocol=protocol, protocol_sha256=sha256(PROTOCOL), code=code_manifest(),
                backbone_weights=dict(sha256=sha256(WEIGHTS), bytes=WEIGHTS.stat().st_size), groups=groups, accepted=accepted)
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
        if progress.get('status') == 'running' and progress.get('phase') == 'training':
            line += (f" epoch {progress.get('epoch')}/{progress.get('total_epochs')} step {progress.get('step')}/{progress.get('total_steps')}"
                     f" loss {progress.get('loss', float('nan')):.4f}")
        elif progress.get('status') == 'running' and progress.get('total_steps'):
            line += f" image {progress.get('step')}/{progress.get('total_steps')}"
        print(line)
    for manifest_path in sorted((OUTPUT/'training').glob('*/seed*/manifest.json')) if (OUTPUT/'training').exists() else []:
        manifest = maybe(manifest_path)
        if manifest.get('complete'):
            print(f"    {manifest_path.parent.parent.name}/{manifest_path.parent.name}: trained in {manifest['active_seconds']/3600:.2f} h, "
                  f"final loss {manifest['history'][-1]['loss']:.4f}")
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
            print('Verifying the backbone weights, the training lists and the test files; sealing the protocol ...', flush=True)
            plan, held = lock_plan(protocol)
            if queue.get('plan_digest') not in (None, plan['digest']):
                queue['jobs'] = {}   # only reachable before any completed training or evaluation (see lock_plan)
            queue['plan_digest'] = plan['digest']
            save()
            from research_polyp_pvt import evaluate as evaluation
            from . import work
            control = Control(CONTROL)
            loaded = {}

            def train_data(group):
                if group not in loaded:
                    loaded.clear()         # one training set in memory at a time
                    print(f'Reading the training images of {group} into memory ...', flush=True)
                    loaded[group] = work.load_data(protocol, group, plan['groups'][group]['train_n'])
                return loaded[group]

            first_group = next(iter(protocol['groups']))

            def check():
                result = work.smoke_test(protocol, first_group, train_data(first_group), plan['groups'][first_group]['train_n'])
                write_json(CONTROL/'check.json', dict(passed=True, checked_at=now(), group=first_group,
                                                      train_n={g: v['train_n'] for g, v in plan['groups'].items()}, **result))
                print(f"Check passed: training images {dict((g, v['train_n']) for g, v in plan['groups'].items())}, "
                      f"peak GPU memory {result['peak_gpu_memory_gb']:.1f} of {result['gpu_total_memory_gb']:.1f} GB", flush=True)
            job('check', check)

            def cohort(group, target):
                if target == protocol['held_out_target']:
                    return held
                folder = ROOT/'data'/target/'Test_Folder'
                records = read_text(ROOT/'data'/target/'Prompts_Folder'/plan['groups'][group]['test_list'])
                if len(records) != plan['groups'][group]['cohorts'][target]['n']:
                    raise ValueError('Unexpected cohort size: '+target)
                return folder, records

            for group, spec in protocol['groups'].items():
                for seed in protocol['seeds']:
                    keys = [f'train/{group}/seed{seed}']+[f'eval/{group}/seed{seed}/{t}' for t in all_targets(protocol, group)]
                    if all(queue['jobs'].get(k, {}).get('status') == 'completed' for k in keys):
                        continue
                    job(f'train/{group}/seed{seed}', lambda g=group, s=seed: work.train(
                        protocol, g, s, plan['digest'], control, train_data(g), plan['groups'][g]['train_n']))
                    if pause.exists():
                        raise Paused()
                    queue['current_job'] = f'eval/{group}/seed{seed}/load'
                    save()
                    model, digest = work.load_model(group, seed, plan['digest'])
                    try:
                        for target in all_targets(protocol, group):
                            folder, records = cohort(group, target)
                            job(f'eval/{group}/seed{seed}/{target}', lambda g=group, s=seed, t=target, f=folder, r=records: evaluation.evaluate(
                                model, f, r, protocol['recipe']['input_size'], protocol['timing']['warmup_images'], evaluation_dir(g, s, t),
                                dict(suite=NAME, plan_digest=plan['digest'], group=g, seed=s, cohort=t, weights_sha256=digest), control, t))
                    finally:
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                from .report import write_report
                write_report(partial=True)       # the report is refreshed after every completed group
                print('GROUP COMPLETE', group, '| report updated', flush=True)
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
            print('PAUSED. Start again with the same command; completed epochs and jobs are kept.', flush=True)
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
