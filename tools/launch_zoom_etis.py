"""Windows entry point for the one-time ETIS-Larib evaluation. Runs in this window so that progress is visible."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-Action', '--action', dest='action', default='Status', choices=['Run', 'Pause', 'Status', 'Report'])
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    environment = {k.upper(): v for k, v in os.environ.items()}
    environment.update(HF_HOME=str(workspace/'.cache/huggingface'), TORCH_HOME=str(workspace/'.cache/torch'),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    return subprocess.call([sys.executable, '-u', '-m', 'research_zoom_etis.runner', '--action', args.action.lower()],
                           cwd=workspace/'MedCLIPSeg', env=environment)


if __name__ == '__main__':
    raise SystemExit(main())
