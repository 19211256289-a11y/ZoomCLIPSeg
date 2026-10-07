"""Queue of the same-protocol Polyp-PVT reference: check, then for every seed training and evaluation, then the report.
Shares the GPU lock with every other queue; can be paused and continued."""
import argparse
import gc
import os
import time
import traceback

from .common import *


def lock_plan(protocol):
    """Seals protocol, code, backbone weights, training data, test cohorts and the accepted results of our models that
    the report compares with. Computes no performance number."""
    from research_zoom_ext.common import Source, seal_cohorts
    from research_zoom_etis import common as etis
    from .data import train_seal
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f'Backbone weights not found: {WEIGHTS}\n    Download {protocol["model"]["backbone_weights_source"].split(" ")[0]} to this path.')
    source = Source(dict(source='kvasir_v1', runs=[]))
    if tuple(source.targets) != TARGETS or source.small != protocol['small_target_threshold']:
        raise ValueError('Test sets or small-target threshold differ from the accepted evaluation')
    cohorts = {t: {k: v for k, v in c.items() if k != 'groups'} for t, c in seal_cohorts(source).items()}
    for target, cohort in cohorts.items():
        if cohort['n'] != COHORT_SIZES[target]:
            raise ValueError('Unexpected cohort size: '+target)
    held_folder, held_records, held_seal = etis.prepare_cohort(etis.load_protocol())
    cohorts[HELD_OUT] = held_seal
    accepted = {}
    for model in protocol['comparison']['ours']:
        for seed in protocol['seeds']:
            for target in TARGETS+(HELD_OUT,):
                path, _ = accepted_cases(protocol, model, seed, target)
                if not path.is_file():
                    raise ValueError(f'Accepted result missing: {path}')
                accepted[f'{model}/seed{seed}/{target}'] = sha256(path)
    plan = dict(protocol=protocol, protocol_sha256=sha256(PROTOCOL), code=code_manifest(),
                backbone_weights=dict(sha256=sha256(WEIGHTS), bytes=WEIGHTS.stat().st_size), train=train_seal(protocol),
                cohorts=cohorts, test_list=source.test_list, accepted=accepted)
    plan['digest'] = object_hash(plan)
    path = OUTPUT/'plan.json'
    if path.exists() and read(path)['digest'] != plan['digest']:
        finished = [p for p in OUTPUT.rglob('manifest.json') if maybe(p).get('complete')]
        if finished:
            raise ValueError('Protocol, code or inputs changed after results of this suite were produced; use a new suite name')
    write_json(path, plan)
    return plan, source, (held_folder, held_records)


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
    for seed_dir in sorted((OUTPUT/'training').glob('seed*')) if (OUTPUT/'training').exists() else []:
        manifest = maybe(seed_dir/'manifest.json')
        if manifest.get('complete'):
            print(f"    {seed_dir.name}: trained in {manifest['active_seconds']/3600:.2f} h, final loss {manifest['history'][-1]['loss']:.4f}")
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
            print('Verifying the backbone weights, the training data and the test files; sealing the protocol ...', flush=True)
            plan, source, held = lock_plan(protocol)
            if queue.get('plan_digest') not in (None, plan['digest']):
                queue['jobs'] = {}   # only reachable before any completed training or evaluation (see lock_plan)
            queue['plan_digest'] = plan['digest']
            save()
            from . import train as training
            from . import evaluate as evaluation
            from .data import load_train
            control = Control(CONTROL)
            data = None

            def train_data():
                nonlocal data
                if data is None:
                    print('Reading the training images into memory ...', flush=True)
                    data = load_train(protocol)
                return data

            def check():
                result = training.smoke_test(protocol, train_data())
                write_json(CONTROL/'check.json', dict(passed=True, checked_at=now(), **result))
                print(f"Check passed: {result['parameters']/1e6:.2f} M parameters, backbone tensors {result['backbone_tensors_loaded']}/"
                      f"{result['backbone_tensors_total']}, peak GPU memory {result['peak_gpu_memory_gb']:.1f} of "
                      f"{result['gpu_total_memory_gb']:.1f} GB, about {result['seconds_per_step_incl_warmup']:.2f} s per step", flush=True)
            job('check', check)

            def cohort(target):
                if target == HELD_OUT:
                    return held
                folder = ROOT/'data'/target/'Test_Folder'
                records = read_text(ROOT/'data'/target/'Prompts_Folder'/plan['test_list'])
                if len(records) != plan['cohorts'][target]['n']:
                    raise ValueError('Unexpected cohort size: '+target)
                return folder, records

            for seed in protocol['seeds']:
                targets = TARGETS+(HELD_OUT,)
                keys = [f'train/seed{seed}']+[f'eval/seed{seed}/{t}' for t in targets]
                if all(queue['jobs'].get(k, {}).get('status') == 'completed' for k in keys):
                    continue
                job(f'train/seed{seed}', lambda s=seed: training.train_seed(protocol, s, plan['digest'], control, train_data()))
                if pause.exists():
                    raise Paused()
                queue['current_job'] = f'eval/seed{seed}/load'
                save()
                model, digest = evaluation.load_model(seed, plan['digest'])
                try:
                    for target in targets:
                        folder, records = cohort(target)
                        job(f'eval/seed{seed}/{target}', lambda t=target, f=folder, r=records: evaluation.evaluate(
                            model, f, r, protocol['recipe']['input_size'], protocol['timing']['warmup_images'], evaluation_dir(seed, t),
                            dict(suite=NAME, plan_digest=plan['digest'], seed=seed, cohort=t, weights_sha256=digest), control, t))
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
