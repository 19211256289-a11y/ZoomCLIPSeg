"""Windows entry point for the control experiments (21 training runs and their evaluations)."""
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
    parser.add_argument('-Action', '--action', dest='action', default='Status', choices=['Run', 'Resume', 'Pause', 'Status', 'Watch', 'Report'])
    parser.add_argument('-Now', '--now', dest='now', action='store_true', help='do not wait for earlier queues')
    parser.add_argument('-Foreground', '--foreground', dest='foreground', action='store_true')
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    root = workspace/'MedCLIPSeg'
    control = workspace/'control/research_controls/formal_controls_kvasir_v1'
    environment = {k.upper(): v for k, v in os.environ.items()}
    environment.update(HF_HOME=str(workspace/'.cache/huggingface'), TORCH_HOME=str(workspace/'.cache/torch'),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    base = [sys.executable, '-u', '-m', 'research_controls.runner', '--action']
    if args.action == 'Watch':
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '(refreshes every 30 s; Ctrl+C only closes this view)\n')
            subprocess.call(base+['status'], cwd=root, env=environment)
            time.sleep(30)
    if args.action not in ('Run', 'Resume'):
        return subprocess.call(base+[args.action.lower()], cwd=root, env=environment)
    command = base+['now' if args.now else 'run']
    if args.foreground:
        return subprocess.call(command, cwd=root, env=environment)
    launcher = control/'launcher.json'
    if launcher.exists():
        previous = json.loads(launcher.read_text(encoding='utf-8'))
        listing = subprocess.run(['tasklist', '/FI', f"PID eq {int(previous['pid'])}", '/NH'], capture_output=True, text=True).stdout.lower()
        if str(int(previous['pid'])) in listing and 'python' in listing:
            print('The control experiments are already running or waiting (PID', str(previous['pid'])+').')
            return 0
    logs = root/'outputs_research/formal_controls_kvasir_v1/queue_logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out, err = logs/f'{stamp}.out.log', logs/f'{stamp}.err.log'
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    with out.open('w', encoding='utf-8') as stdout, err.open('w', encoding='utf-8') as stderr:
        child = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                 startupinfo=startup, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    control.mkdir(parents=True, exist_ok=True)
    launcher.write_text(json.dumps(dict(pid=child.pid, out=str(out), err=str(err), started=stamp)), encoding='utf-8')
    print('Control experiments started in the background, PID', child.pid)
    print('They wait for earlier queues and for the GPU, then run by themselves.')
    print('Progress:  ResearchControls.cmd -Action Status   (or -Action Watch)')
    print('Errors:', err)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
