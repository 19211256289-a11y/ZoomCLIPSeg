"""Windows entry point for the extension campaign (BUSI suite, then two additional Kvasir seeds)."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-Action', '--action', dest='action', default='Status',
                        choices=['Preflight', 'Run', 'Resume', 'Pause', 'Status', 'Watch', 'Report'])
    parser.add_argument('-Foreground', '--foreground', dest='foreground', action='store_true')
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    root = workspace/'MedCLIPSeg'
    control = workspace/'control/research_detail_ext'
    environment = {k.upper(): v for k, v in os.environ.items()}
    environment.update(HF_HOME=str(workspace/'.cache/huggingface'), TORCH_HOME=str(workspace/'.cache/torch'),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    command = [sys.executable, '-u', '-m', 'research_detail_ext.runner', '--action']
    queue = control/'queue.json'
    state = lambda: json.loads(queue.read_text(encoding='utf-8')).get('state') if queue.exists() else None
    if args.action == 'Watch':
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '(refreshes every 30 s; Ctrl+C only closes this view)\n')
            subprocess.call(command+['status'], cwd=root, env=environment)
            if state() in ('completed', 'failed', 'paused'):
                return 0
            time.sleep(30)
    if args.action not in ('Run', 'Resume') or args.foreground:
        return subprocess.call(command+[args.action.lower()], cwd=root, env=environment)
    if state() == 'completed':
        print('Already completed. See reports/formal_detail_mix_busi_v1_results and reports/formal_detail_mix_kvasir_v2_results')
        return 0
    if state() in ('paused', 'failed', 'running'):
        # Finished jobs are kept. Training continues from its last saved checkpoint.
        args.action = 'Resume'
    logs = root/'outputs_research/detail_ext_logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out, err = logs/f'{stamp}.out.log', logs/f'{stamp}.err.log'
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    with out.open('w', encoding='utf-8') as stdout, err.open('w', encoding='utf-8') as stderr:
        child = subprocess.Popen(command+[args.action.lower()], cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=stdout, stderr=stderr, startupinfo=startup,
                                 creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    control.mkdir(parents=True, exist_ok=True)
    (control/'launcher.json').write_text(json.dumps(dict(pid=child.pid, out=str(out), err=str(err), started=stamp)), encoding='utf-8')
    print('Campaign started in the background, PID', child.pid)
    print('Progress:  ResearchDetailExt.cmd -Action Watch   (or -Action Status)')
    print('Log:', out)
    print('Errors:', err)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
