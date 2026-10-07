"""Windows entry point: export the masks of the accepted two-pass evaluation for figures (waits for the GPU)."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-Action', '--action', dest='action', default='Status', choices=['Run', 'Pause', 'Status'])
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    root = workspace/'MedCLIPSeg'
    environment = {k.upper(): v for k, v in os.environ.items()}
    environment.update(HF_HOME=str(workspace/'.cache/huggingface'), TORCH_HOME=str(workspace/'.cache/torch'),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    command = [sys.executable, '-u', '-m', 'research_zoom_export.runner', '--action', args.action.lower()]
    if args.action != 'Run':
        return subprocess.call(command, cwd=root, env=environment)
    logs = root/'outputs_research/zoom_export_v1/logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out, err = logs/f'{stamp}.out.log', logs/f'{stamp}.err.log'
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    with out.open('w', encoding='utf-8') as stdout, err.open('w', encoding='utf-8') as stderr:
        child = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                 startupinfo=startup, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    print('Mask export started in the background, PID', child.pid)
    print('It waits for the GPU and then needs about ten minutes.')
    print('Progress:  ExportZoomMasks.cmd -Action Status')
    print('Errors:', err)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
