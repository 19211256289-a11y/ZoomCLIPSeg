"""Queue: Step 0 analysis, protocol lock, Val reproduction check, 60 Test/OOD evaluations, report."""
from local_runtime import configure_local_models
configure_local_models()
import argparse
import gc
import os
import time
import traceback
from pathlib import Path

from .common import *

N_JOBS = 2+len(RUNS)*(1+len(TARGETS))
N_IMAGES = len(RUNS)*sum(COHORT_SIZES.values())


def lock_plan(protocol, accepted):
    """Seal checkpoints and input files. Computes no performance number."""
    from research_formal import common as base
    from research_formal.metrics import foreground_threshold
    from research_detail_formal.common import load_suite, make_config
    if foreground_threshold(make_config(load_suite(), RUNS[0])) != SMALL:
        raise ValueError('Source small-target threshold differs from the locked value')
    cohorts = base.test_cohort_manifest(base.load_suite('formal_kvasir_v1'))
    earlier = read(ROOT/'outputs_research/formal_kvasir_v1/experiment_plan.json')['test_cohorts']
    for target in TARGETS:
        if COHORT_SIZES[target] != cohorts[target]['n']:
            raise ValueError('Unexpected cohort size: '+target)
        for key in ('n', 'prompt_sha256', 'files'):
            if cohorts[target][key] != earlier[target][key]:
                raise ValueError(f'{target}: test inputs differ from the cohort sealed for the earlier formal study')
    checkpoints = {}
    for run in RUNS:
        path = checkpoint_path(run)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints[run] = dict(path=str(path), bytes=path.stat().st_size, accepted_sha256=accepted[run]['checkpoint_sha256'])
    plan = dict(protocol=protocol, protocol_sha256=sha256(PROTOCOL), code=code_manifest(), checkpoints=checkpoints,
                test_cohorts=cohorts, small_threshold=SMALL)
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
    done = [k for k, j in jobs.items() if j.get('status') == 'completed']
    print('Suite:', NAME)
    print('Queue:', queue.get('state', 'not_started'), '| Pause requested:', (CONTROL/'PAUSE.request').exists())
    print(f'Jobs completed: {len(done)}/{N_JOBS}')
    evaluated = [(k, jobs[k]) for k in done if '/' in k]
    images = sum(COHORT_SIZES[k.split('/')[1]] for k, _ in evaluated)
    current = queue.get('current_job')
    line = f'Images evaluated: {images}/{N_IMAGES}'
    if queue.get('state') == 'running' and current and progress.get('status') == 'running' and '/' in current \
            and f"{progress.get('run_id')}/{progress.get('cohort')}" == current:
        line += f" | Current: {current}, batch {progress.get('step')}/{progress.get('total_steps')}"
        images += progress.get('cases_completed', 0)
    elif queue.get('state') == 'running':
        line += f' | Current: {current}'
    print(line)
    seconds = sum(j.get('active_seconds', 0) for _, j in evaluated)
    rate = seconds/max(1, sum(COHORT_SIZES[k.split('/')[1]] for k, _ in evaluated)) if evaluated else .6
    loads = len(RUNS)-sum(all(jobs.get(f'{r}/{t}', {}).get('status') == 'completed' for t in TARGETS) for r in RUNS)
    if queue.get('state') not in ('completed',):
        print(f'Estimated remaining: {((N_IMAGES-images)*rate+loads*45)/60:.0f} min (excludes pauses)')
    for run in RUNS:
        check = maybe(OUTPUT/'evaluations'/run/'Val_check/comparison.json')
        if check:
            print(f"  Val check {run}: local {check['local_DSC_percent']:.4f} vs accepted {check['accepted_DSC_percent']:.4f} ({check['difference_pp']:+.4f} pp)")
    if queue.get('error'):
        print('ERROR:', queue['error'])
    if queue.get('state') == 'completed':
        print('Report:', REPORT/'report.md')
    print('Step 0 report:', STEP0/'report.md', '(exists)' if (STEP0/'report.md').exists() else '(not yet)')
    print('Updated:', queue.get('updated_at', 'n/a'), '(UTC)')
    return queue.get('state', 'not_started')


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
    with acquire_gpu_queue():
        path = CONTROL/'queue.json'
        queue = maybe(path) or dict(jobs={})
        queue.update(state='running', pid=os.getpid(), error=None, started_at=queue.get('started_at', now()))

        def save():
            queue['updated_at'] = now()
            write_json(path, queue)

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
            accepted = accepted_runs(protocol)
            from .step0 import main as step0
            job('step0', step0)
            queue['current_job'] = 'lock'
            save()
            plan = lock_plan(protocol, accepted)
            queue['plan_digest'] = plan['digest']
            save()
            from .evaluate import load_model, evaluate_cohort
            tolerance = protocol['val_reproduction_check']['abort_if_abs_dsc_difference_pp_exceeds']
            for run in RUNS:
                keys = [f'{run}/Val_check']+[f'{run}/{target}' for target in TARGETS]
                if all(queue['jobs'].get(k, {}).get('status') == 'completed' for k in keys):
                    continue
                if pause.exists():
                    raise TrainingPaused()
                queue['current_job'] = f'{run}/load'
                save()
                model, cfg, digest = load_model(run, accepted[run]['checkpoint_sha256'])
                control = TrainingControl(str(CONTROL), run_id=run, phase='evaluation')
                base = dict(run=run, checkpoint_sha256=digest, plan_digest=plan['digest'], prompt=GENERIC,
                            mc_samples=30, batch_size=32, threshold=.5)

                def val_check():
                    text = Path(cfg.DATASET.TEXT_PROMPT_PATH)/'Val_text.xlsx'
                    destination = OUTPUT/'evaluations'/run/'Val_check'
                    summary = evaluate_cohort(model, cfg, 'Val_check', cfg.DATASET.VAL_PATH, read_text(text), destination,
                                              dict(base, cohort='Val_check', list_sha256=sha256(text)), control, save_maps=False)
                    expected = accepted[run]['summary']
                    difference = summary['DSC_percent']-expected['DSC_percent']
                    write_json(destination/'comparison.json', dict(
                        run=run, accepted_DSC_percent=expected['DSC_percent'], local_DSC_percent=summary['DSC_percent'],
                        difference_pp=difference, accepted_small_DSC_percent=expected['small_DSC_percent'],
                        local_small_DSC_percent=summary['small_DSC_percent'], tolerance_pp=tolerance,
                        passed=abs(difference) <= tolerance, device=summary['device']))
                    if summary['n'] != 100 or summary['n_small'] != 25:
                        raise ValueError('Unexpected Val cohort')
                    if abs(difference) > tolerance:
                        raise RuntimeError(f'{run}: local Val DSC differs from the accepted value by {difference:+.4f} pp; '
                                           'Test evaluation was not started for this run')

                def test(target):
                    cohort = plan['test_cohorts'][target]
                    folder = ROOT/'data'/target/'Test_Folder'
                    text = ROOT/'data'/target/'Prompts_Folder'/protocol['test_list']
                    if sha256(text) != cohort['prompt_sha256']:
                        raise ValueError('Test list changed after locking')
                    evaluate_cohort(model, cfg, target, folder, read_text(text), OUTPUT/'evaluations'/run/target,
                                    dict(base, cohort=target, split='Test', list_sha256=cohort['prompt_sha256'],
                                         files_digest=object_hash(cohort['files'])), control,
                                    groups=list(cohort['image_hash_groups'].values()))

                job(f'{run}/Val_check', val_check)
                for target in TARGETS:
                    job(f'{run}/{target}', lambda t=target: test(t))
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
            print('PAUSED. Use ResearchDetailTest.cmd -Action Resume.', flush=True)
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
    parser.add_argument('--action', choices=['step0', 'run', 'resume', 'pause', 'status', 'report'], default='status')
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
    if args.action == 'step0':
        from .step0 import main as step0
        step0()
        return 0
    if args.action == 'report':
        from .report import write_report
        print(write_report())
        return 0
    return work(args.action)


if __name__ == '__main__':
    raise SystemExit(main())
