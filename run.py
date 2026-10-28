"""Run one experiment package in the foreground, with the environment the original launchers set.

    python run.py <package> <action> [extra runner arguments]

Examples:
    python run.py research_zoom_etis status
    python run.py research_zoom_etis run
    python run.py research_zoom_ext run --suites zoom_busi_v1
    python run.py research_formal check --suite formal_kvasir_v1

The runner of each package is `MedCLIPSeg/<package>/runner.py`; `python run.py <package> --help` lists its actions.
Runs are offline: models are read from `.cache/` and `MedCLIPSeg/checkpoints/` (see docs/SETUP.md).
"""
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
ROOT = WORKSPACE/'MedCLIPSeg'


def main(argv):
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        packages = sorted(p.name for p in ROOT.glob('research*') if (p/'runner.py').exists())
        print('Packages:', ', '.join(packages))
        return 0
    package, rest = argv[0], argv[1:]
    if not (ROOT/package/'runner.py').exists():
        print(f'No runner for {package!r}; run without arguments for the list.', file=sys.stderr)
        return 2
    if rest and not rest[0].startswith('-'):
        rest = ['--action', rest[0]] + rest[1:]
    environment = dict(os.environ)
    environment.update(HF_HOME=str(WORKSPACE/'.cache/huggingface'), TORCH_HOME=str(WORKSPACE/'.cache/torch'),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    return subprocess.call([sys.executable, '-u', '-m', f'{package}.runner'] + rest, cwd=ROOT, env=environment)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
