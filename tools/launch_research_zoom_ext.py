"""Windows entry point for the follow-up two-pass zoom evaluations (no training)."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DEFAULT_RUN = 'zoom_busi_v1,zoom_ablation_kvasir_v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-Action', '--action', dest='action', default='Status',
                        choices=['Run', 'Resume', 'Pause', 'Status', 'Watch', 'Report'])
    parser.add_argument('-Suites', '--suites', dest='suites', default=None,
                        help='comma-separated suite names; Run uses "'+DEFAULT_RUN+'" when omitted')
    parser.add_argument('-Now', '--now', dest='now', action='store_true',
                        help='do not wait for the local training queue (BUSI) to finish')
    parser.add_argument('-Foreground', '--foreground', dest='foreground', action='store_true')
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    root = workspace/'MedCLIPSeg'
    control = workspace/'control/research_zoom_ext'
    environment = {k.upper(): v for k, v in os.environ.items()}
    environment.update(HF_HOME=str(workspace/'.cache/huggingface'), TORCH_HOME=str(workspace/'.cache/torch'),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    base = [sys.executable, '-u', '-m', 'research_zoom_ext.runner']
    if args.action == 'Resume':
        args.action = 'Run'   # starting again continues every unfinished suite and clears pause requests
    if args.action == 'Watch':
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '(refreshes every 30 s; Ctrl+C only closes this view)\n')
            subprocess.call(base+['--action', 'status', '--suites', args.suites or 'all'], cwd=root, env=environment)
            time.sleep(30)
    if args.action != 'Run':
        return subprocess.call(base+['--action', args.action.lower(), '--suites', args.suites or 'all'], cwd=root, env=environment)
    suites = args.suites or DEFAULT_RUN
    command = base+['--action', 'run', '--suites', suites]+(['--now'] if args.now else [])
    if args.foreground:
        return subprocess.call(command, cwd=root, env=environment)
    launcher = control/'launcher.json'
    if launcher.exists():
        previous = json.loads(launcher.read_text(encoding='utf-8'))
        listing = subprocess.run(['tasklist', '/FI', f"PID eq {int(previous['pid'])}", '/NH'], capture_output=True, text=True).stdout.lower()
        if str(int(previous['pid'])) in listing and 'python' in listing:
            print('A zoom evaluation process is already running (PID', str(previous['pid'])+').')
            print('Use -Action Status, or -Action Pause and start again once it has stopped.')
            return 1
    logs = root/'outputs_research/zoom_ext_logs'
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
    launcher.write_text(json.dumps(dict(pid=child.pid, suites=suites, out=str(out), err=str(err), started=stamp)), encoding='utf-8')
    print('Started in the background, PID', child.pid, '| suites in order:', suites)
    print('Each suite waits for its inputs and for the GPU, then runs by itself.')
    print('Progress:  ResearchZoomExt.cmd -Action Watch   (or -Action Status)')
    print('Log:', out)
    print('Errors:', err)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
