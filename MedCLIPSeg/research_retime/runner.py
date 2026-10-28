"""Repeats the timing of text_probe_v1 (one inference run of detail_seed666 with 1, 5, 15 and 30 stochastic passes)
three times in one session, to settle the difference between two earlier measurements of the 30-pass run (0.76 s in
zoom_probe_v1, 0.94 s in text_probe_v1). Timing only; no accuracy is computed. Shares the GPU lock."""
from local_runtime import configure_local_models
configure_local_models()
import argparse
import os
import time
import traceback

import yaml

from research_formal.common import ROOT, WORKSPACE, now, write_json
from research_detail_test.common import read, maybe, TARGETS

NAME = 'retime_v1'
CONTROL = WORKSPACE/'control/research_retime'
OUTPUT = ROOT/'outputs_research'/NAME
REPORT = WORKSPACE/'reports'/f'{NAME}_results'
RUN, REPEATS = 'detail_seed666', 3


def work():
    import numpy as np
    import torch
    from local_reproduce import acquire_gpu_queue
    from train_detail_formal import TrainingControl
    from utils.main_utils import read_text
    CONTROL.mkdir(parents=True, exist_ok=True)
    pause, path = CONTROL/'PAUSE.request', CONTROL/'queue.json'
    pause.unlink(missing_ok=True)
    queue = maybe(path) or dict(jobs={})
    if queue.get('state') == 'completed':
        print(NAME, 'already completed. Report:', REPORT/'report.md', flush=True)
        return 0
    queue.update(n_jobs=REPEATS, error=None, pid=os.getpid())

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
        save()
        try:
            from research_zoom_ext.common import Source
            import research_zoom.evaluate as zoom_evaluate
            import research_detail_test.evaluate as test_evaluate
            from research_text_probe import evaluate as probe
            from research_text_probe.common import PROTOCOL
            protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
            source = Source(dict(source='kvasir_v1', runs=[RUN]))
            zoom_evaluate.GENERIC = test_evaluate.GENERIC = source.prompt
            model, cfg, digest = source.load_model(RUN, source.expected_checkpoints()[RUN])
            folder = ROOT/'data'/TARGETS[0]/'Test_Folder'
            records = read_text(ROOT/'data'/TARGETS[0]/'Prompts_Folder'/source.test_list)
            tracker = TrainingControl(str(CONTROL), run_id=RUN, phase='evaluation')
            for repeat in range(1, REPEATS+1):
                key = f'timing/{repeat}'
                if queue['jobs'].get(key, {}).get('status') == 'completed':
                    continue
                queue['current_job'] = key
                save()
                probe.timing(model, cfg, folder, records, source.prompt, protocol, OUTPUT/f'timing_{repeat}.json', tracker)
                queue['jobs'][key] = dict(status='completed', completed_at=now())
                save()
                print('COMPLETE', key, flush=True)
            runs = [read(OUTPUT/f'timing_{r}.json') for r in range(1, REPEATS+1)]
            passes = list(runs[0]['passes'])
            summary = {k: dict(mean_ms_per_repeat=[r['passes'][k]['mean_ms'] for r in runs],
                               mean_ms=float(np.mean([r['passes'][k]['mean_ms'] for r in runs]))) for k in passes}
            REPORT.mkdir(parents=True, exist_ok=True)
            write_json(REPORT/'results.json', dict(name=NAME, created_at=now(), run=RUN, checkpoint_sha256=digest, device=runs[0]['device'],
                                                   torch=runs[0]['torch'], repeats=REPEATS, passes=summary, note=runs[0]['note']))
            lines = [f'# {NAME}: time of one inference run ({RUN}, {runs[0]["device"]}, batch size 1), three repetitions', '',
                     '| stochastic passes | repeat 1 (ms) | repeat 2 (ms) | repeat 3 (ms) | mean (ms) |', '|---|---|---|---|---|']
            lines += [f"| {k} | "+' | '.join(f'{v:.0f}' for v in summary[k]['mean_ms_per_repeat'])+f" | {summary[k]['mean_ms']:.0f} |" for k in passes]
            lines += ['', 'Earlier measurements of the 30-pass run: 760 ms (zoom_probe_v1, 2026-10-06) and 942 ms (text_probe_v1, 2026-10-07).']
            (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
            queue.update(state='completed', completed_at=now(), current_job=None)
            save()
            print('ALL COMPLETE. Report:', REPORT/'report.md', flush=True)
            return 0
        except BaseException as exc:
            paused = type(exc).__name__ == 'TrainingPaused'
            queue.update(state='paused' if paused else 'failed', error=None if paused else f'{type(exc).__name__}: {exc}')
            save()
            if paused:
                print('PAUSED. Start again with the same command.', flush=True)
                return 75
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['run', 'pause', 'status', 'report'], default='status')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'pause':
        CONTROL.mkdir(parents=True, exist_ok=True)
        (CONTROL/'PAUSE.request').write_text(now(), encoding='utf-8')
        print('Pause requested.')
        return 0
    if args.action in ('status', 'report'):
        queue = maybe(CONTROL/'queue.json')
        print(f"{NAME}: {queue.get('state', 'not_started')} | current: {queue.get('current_job')} | updated: {queue.get('updated_at', 'n/a')} (UTC)")
        if queue.get('error'):
            print('    ERROR:', queue['error'])
        return 0
    return work()


if __name__ == '__main__':
    raise SystemExit(main())
