"""Windows-friendly entry point; hidden background prompt training with safe pause/resume."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-Action','--action',dest='action',choices=['Prepare','Check','Train','Pause','Resume','Status','Report'],default='Status')
    parser.add_argument('-Foreground','--foreground',dest='foreground',action='store_true')
    args=parser.parse_args()
    workspace=Path(__file__).resolve().parents[1]
    migration=workspace/'control/research_detail_formal/formal_detail_mix_kvasir_v1/local/REMOTE_EXECUTOR.json'
    if args.action in ('Train','Resume') and migration.exists():
        raise SystemExit('Seed666 execution has moved to the 5090 server. Use ResearchDetail5090.cmd -Action Resume. Local checkpoints are preserved; do not resume a duplicate local queue.')
    root=workspace/'MedCLIPSeg'
    environment={k.upper():v for k,v in os.environ.items()}
    environment.update(HF_HOME=str(workspace/'.cache/huggingface'),TORCH_HOME=str(workspace/'.cache/torch'),
                       HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
                       OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    command=[sys.executable,'-u','-m','research_detail_formal.runner','--action',args.action.lower()]
    policy=workspace/'control/research_detail_formal/formal_detail_mix_kvasir_v1/local/execution_policy.json'
    if policy.exists() and args.action in ('Train','Resume','Pause','Status'):
        import json
        if json.loads(policy.read_text(encoding='utf-8')).get('mode')!='two_experiment_lanes':
            raise ValueError('Unknown execution policy')
        command=[sys.executable,'-u',str(workspace/'tools/run_detail_formal_parallel.py'),'--action',args.action.lower()]
    if args.action not in ('Train','Resume') or args.foreground:
        return subprocess.call(command,cwd=root,env=environment)
    logs=root/'outputs_research/formal_detail_mix_kvasir_v1/local/logs'
    logs.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out,err=logs/f'{stamp}.out.log',logs/f'{stamp}.err.log'
    startup=subprocess.STARTUPINFO()
    startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow=subprocess.SW_HIDE
    with out.open('w',encoding='utf-8') as stdout,err.open('w',encoding='utf-8') as stderr:
        child=subprocess.Popen(command,cwd=root,env=environment,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                               startupinfo=startup,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    print('Query module pilot launched, PID',child.pid,'; use ResearchDetailFormal.cmd -Action Status to confirm startup.')
    print('Log:',out)
    print('Errors:',err)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
