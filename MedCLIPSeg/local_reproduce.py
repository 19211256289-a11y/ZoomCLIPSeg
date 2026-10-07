"""Run the author's 24 ID + 10 OOD experiments with an auditable status ledger.

Run from this repository using the workspace Python/environment launcher.
This program does not download or invent missing datasets/checkpoints.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from local_runtime import configure_local_models
from local_data import dataset_readiness

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
ID_DATASETS = ['BUSI', 'BTMRI', 'ISIC', 'Kvasir', 'Covid19', 'EUS']
TARGETS = {'BUSI': ['BUSBRA', 'BUSUC', 'BUID', 'UDIAT'],
           'Kvasir': ['ColonDB', 'ClinicDB', 'CVC300', 'BKAI'],
           'BTMRI': ['BRISC'], 'ISIC': ['UWaterlooSkinCancer']}
CKPT = 'MedCLIPSeg_unimedclip_ViT-B-16_latest.pth'
CONTROL = WORKSPACE/'control'


def acquire_gpu_queue():
    """An OS lock survives stale PID files and closes automatically after a crash."""
    handle = (CONTROL/'gpu_queue.lock').open('a+b')
    if handle.tell() == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('A reproduction queue is already running; use Pause/Resume or inspect control/coordinator.json.')
    return handle


def write_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def last_log_error(path):
    """Include the final exception line, instead of only the subprocess exit code."""
    if path is None or not path.is_file():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 8192))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    return next((line.strip() for line in reversed(lines) if line.strip()), None)


def wait_until_resumed(ledger, ledger_path):
    while (CONTROL/'PAUSE.request').exists():
        if ledger.get('state') != 'paused':
            ledger['state'] = 'paused'
            ledger['paused_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            write_json(ledger_path, ledger)
            print('PAUSED: GPU child released. Run Resume-Training.cmd to continue.', flush=True)
        time.sleep(1)
    if ledger.get('state') == 'paused':
        ledger['state'] = 'running'
        write_json(ledger_path, ledger)
        print('RESUMED', flush=True)


def main():
    configure_local_models()
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['eval', 'train'], default='eval')
    p.add_argument('--datasets', nargs='+', default=ID_DATASETS)
    p.add_argument('--percentages', nargs='+', type=int, choices=[10, 25, 50, 100], default=[10, 25, 50, 100])
    p.add_argument('--seed', type=int, default=666)
    p.add_argument('--test-batch-size', type=int, default=32)
    p.add_argument('--output-dir')
    p.add_argument('--include-ood', action='store_true')
    p.add_argument('--rerun', action='store_true')
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--wait-for-data', type=int, default=0, metavar='SECONDS',
                   help='Wait for verified dataset preparation before reporting an unavailable experiment.')
    args = p.parse_args()
    if args.include_ood and 100 not in args.percentages:
        p.error('--include-ood requires 100 in --percentages.')
    os.chdir(ROOT)
    CONTROL.mkdir(exist_ok=True)
    queue_lock = acquire_gpu_queue()
    coordinator = {'pid': os.getpid(), 'mode': args.mode, 'state': 'running',
                   'python': sys.executable, 'argv': sys.argv, 'workspace': str(WORKSPACE),
                   'started_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    write_json(CONTROL/'coordinator.json', coordinator)
    output = args.output_dir or ('outputs_medclipseg' if args.mode == 'eval' else 'outputs_local_training')
    if args.mode == 'train' and Path(output).resolve() == (ROOT/'outputs_medclipseg').resolve():
        p.error('Local training must use a separate directory from official checkpoints.')
    label = Path(output).name
    ledger_path = WORKSPACE/'reports'/f'{label}_{args.mode}_status.json'
    experiments = [(d, d, n) for d in args.datasets for n in args.percentages]
    if args.include_ood and 100 in args.percentages:
        # Evaluate domain transfer after each source finishes, so partial runs are useful.
        experiments = []
        for source in args.datasets:
            experiments += [(source, source, n) for n in args.percentages]
            experiments += [(source, target, 100) for target in TARGETS.get(source, [])]
    ledger = json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else {'experiments': {}}
    signature = {'output_dir': str(Path(output).resolve()), 'seed': args.seed,
                 'test_batch_size': args.test_batch_size,
                 'gradient_checkpointing': args.gradient_checkpointing, 'mode': args.mode}
    if ledger.get('run_signature') and ledger['run_signature'] != signature:
        raise RuntimeError('Run settings differ from this ledger. Use a new --output-dir to preserve experiment provenance.')
    if ledger.get('seed', args.seed) != args.seed or ledger.get('test_batch_size', args.test_batch_size) != args.test_batch_size:
        raise RuntimeError('Existing ledger uses another seed or batch size; choose another --output-dir.')
    ledger['run_signature'] = signature
    ledger['planned_experiments'] = [f'{s}_to_{t}_{n}' for s,t,n in experiments]
    # Keep previous failure evidence, but show unfinished work as queued on restart.
    for source, target, percent in experiments:
        key = f'{source}_to_{target}_{percent}'
        previous = ledger['experiments'].setdefault(key, {
            'source': source, 'target': target, 'percentage': percent, 'status': 'queued'})
        if previous.get('status') != 'completed' or args.rerun:
            if previous.get('status') not in ('queued', None):
                history = previous.setdefault('history', [])
                history.append({k: v for k, v in previous.items() if k != 'history'})
            previous['status'] = 'queued'
    ledger.pop('last_error', None)
    ledger.update(mode=args.mode, python=sys.executable, seed=args.seed, output_dir=output,
                  test_batch_size=args.test_batch_size, pid=os.getpid(),
                  started_at=time.strftime('%Y-%m-%dT%H:%M:%S'), state='running')
    write_json(ledger_path, ledger)
    try:
        subprocess.run([sys.executable, str(WORKSPACE/'tools'/'summarize_reproduction.py')],
                       cwd=WORKSPACE, check=True)
        for source, target, percent in experiments:
            if args.mode == 'train':
                wait_until_resumed(ledger, ledger_path)
            key = f'{source}_to_{target}_{percent}'
            previous = ledger['experiments'].get(key, {})
            if previous.get('status') == 'completed' and Path(previous.get('metrics_csv', '')).is_file() and not args.rerun:
                print(f'SKIP completed: {key}', flush=True)
                continue
            record = {'source': source, 'target': target, 'percentage': percent,
                      'history': previous.get('history', []),
                      'status': 'running', 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
            ledger['experiments'][key] = record
            ledger['current_experiment'] = key
            write_json(ledger_path, ledger)
            config = f'configs/{target}.yaml'
            folder = source if percent == 100 else f'{source}_{percent}'
            checkpoint = Path(output)/folder/'trained_models'/f'seed{args.seed}'/CKPT
            prompts = ROOT/'data'/target/'Prompts_Folder'/'Test_text_original.xlsx'
            images = ROOT/'data'/target/'Test_Folder'/'img'
            common = ['--config-file', config, '--output-dir', output,
                      '--data_percentage', str(percent), '--seed', str(args.seed)]
            commands = []
            if args.mode == 'train' and source == target:
                command = [sys.executable, '-u', 'train.py', *common, '--num-workers', '0']
                if checkpoint.exists():
                    command.append('--resume')
                if args.gradient_checkpointing:
                    command.extend(['--gradient-checkpointing'])
                commands.append(command)
            commands += [[sys.executable, '-u', 'test.py', *common, '--source_dataset', source,
                          '--batch-size', str(args.test_batch_size)],
                         [sys.executable, '-u', 'utils/eval.py', *common]]
            if args.mode == 'train':
                commands[-2].extend(['--control-dir', str(CONTROL)])
            log_path = None
            record['stage'] = 'data_check'
            try:
                if shutil.disk_usage(WORKSPACE).free < 25*2**30:
                    raise RuntimeError('Less than 25 GiB free disk space.')
                deadline = time.monotonic() + args.wait_for_data
                while True:
                    readiness = dataset_readiness(target, percent, args.mode == 'train' and source == target)
                    record['data_readiness'] = readiness
                    if readiness.get('blocked_reason'):
                        raise RuntimeError(f'Dataset preparation blocked: {readiness["blocked_reason"]}')
                    if readiness['ready']:
                        ledger['state'] = 'running'
                        record['status'] = 'running'
                        write_json(ledger_path, ledger)
                        break
                    if time.monotonic() >= deadline:
                        raise FileNotFoundError(f'Dataset not ready: {target}: {readiness["errors"]}')
                    ledger['state'] = 'waiting_for_data'
                    record['status'] = 'waiting_for_data'
                    write_json(ledger_path, ledger)
                    print(f'WAITING {key}: {readiness["errors"]}', flush=True)
                    for _ in range(30):
                        if args.mode == 'train':
                            wait_until_resumed(ledger, ledger_path)
                        time.sleep(1)
                if (args.mode == 'eval' or source != target) and not checkpoint.exists():
                    raise FileNotFoundError(f'Source checkpoint not ready: {checkpoint}')
                log_path = WORKSPACE/'logs'/f'{label}_{key}.log'
                record.update(log=str(log_path), commands=commands)
                write_json(ledger_path, ledger)
                started = time.perf_counter()
                with log_path.open('a', encoding='utf-8') as log:
                    for command in commands:
                        record['stage'] = {'train.py': 'training', 'test.py': 'testing',
                                           'utils/eval.py': 'metrics'}[command[2]]
                        write_json(ledger_path, ledger)
                        while True:
                            if args.mode == 'train':
                                wait_until_resumed(ledger, ledger_path)
                            print(f'RUN {key}: {subprocess.list2cmdline(command)}', flush=True)
                            log.write('\nCOMMAND '+subprocess.list2cmdline(command)+'\n')
                            log.flush()
                            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                            if result.returncode == 75 and any(name in command for name in ['train.py', 'test.py']):
                                # The trainer saved at a safe batch boundary and released the GPU.
                                if 'train.py' in command and checkpoint.exists() and '--resume' not in command:
                                    command.insert(command.index('--num-workers'), '--resume')
                                wait_until_resumed(ledger, ledger_path)
                                continue
                            result.check_returncode()
                            break
                result_folder = target if percent == 100 else f'{target}_{percent}'
                metrics_path = Path(output)/result_folder/'seg_results'/f'seed{args.seed}'/'test_Prompt-original.csv'
                record.update(status='completed', seconds=time.perf_counter()-started,
                              metrics_csv=str(metrics_path.resolve()))
                with metrics_path.open(encoding='utf-8-sig') as stream:
                    rows = list(csv.DictReader(stream))
                record['metrics_rows'] = len(rows)
                if not rows:
                    raise RuntimeError('Evaluation returned zero samples.')
                record['DSC_percent'] = 100*sum(float(r['DSC']) for r in rows)/len(rows)
                record['NSD_percent'] = 100*sum(float(r['NSD']) for r in rows)/len(rows)
                print(f'COMPLETED {key}: {metrics_path}', flush=True)
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                if isinstance(exc, subprocess.CalledProcessError):
                    record['log_error'] = last_log_error(log_path)
                ledger['state'] = 'failed'
                ledger['last_error'] = {'experiment': key, 'stage': record['stage'],
                                        'message': record.get('log_error') or record['error'],
                                        'log': record.get('log')}
                print(f'FAILED {key}: {exc}', flush=True)
                print('QUEUE STOPPED: fix the error, then run Resume-Training.cmd. '
                      'Completed results and saved checkpoints are preserved.', flush=True)
            record['ended_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            write_json(ledger_path, ledger)
            subprocess.run([sys.executable, str(WORKSPACE/'tools'/'summarize_reproduction.py')],
                           cwd=WORKSPACE, check=True)
            if ledger['state'] == 'failed':
                break
        else:
            ledger['state'] = 'completed'
    except KeyboardInterrupt:
        ledger['state'] = 'interrupted'
        raise
    except Exception as exc:
        ledger['state'] = 'failed'
        ledger['last_error'] = {'experiment': ledger.get('current_experiment'),
                                'message': f'{type(exc).__name__}: {exc}'}
        raise
    finally:
        ledger['ended_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        write_json(ledger_path, ledger)
        coordinator.update(state=ledger['state'], ended_at=ledger['ended_at'])
        write_json(CONTROL/'coordinator.json', coordinator)
        queue_lock.close()
    if ledger['state'] != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    main()
