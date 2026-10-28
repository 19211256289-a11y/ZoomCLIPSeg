"""Read-only, CPU-only acceptance of all nine final training/Val runs.

Writes evidence only beside this script. The live evaluation queue is never changed.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv, gc, hashlib, json, re, sys
import cv2
import numpy as np
import torch
cv2.setNumThreads(1)
torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ROOT = WORKSPACE / 'MedCLIPSeg'
OUT = ROOT / 'outputs_research/formal_kvasir_v1'
CONTROL = WORKSPACE / 'control/research_formal/formal_kvasir_v1'
sys.path.insert(0, str(ROOT))
from research_formal.metrics import case_metrics, summarize
from utils.main_utils import read_text

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def no_digest(value):
    return {k: v for k, v in value.items() if k != 'digest'}

def numeric_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    string_keys = {'name', 'image_name', 'mask_file', 'probability_file'}
    return [{k: v if k in string_keys else float(v) if v else None for k, v in row.items()} for row in rows]

def equal_summary(actual, expected):
    for key, value in actual.items():
        if value is None:
            assert expected[key] is None, key
        else:
            assert np.allclose(value, expected[key], atol=1e-8, rtol=1e-6), key

def fingerprint(state, trainable):
    h = hashlib.sha256()
    for name, value in state.items():
        if name.startswith(('pvl_adapters.', 'mask_head.', 'upscale.')) != trainable:
            continue
        value = value.contiguous()
        h.update(f'{name}|{value.dtype}|{list(value.shape)}'.encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()

plan = read(OUT/'experiment_plan.json')
queue_start = read(CONTROL/'queue.json')
assert object_hash(no_digest(plan)) == plan['digest'] == queue_start['plan_digest']
variants = ['paired_repeat', 'paired_image', 'paired_aug']
seeds = [666, 667, 668]
runs = [f'{v}_seed{s}' for s in seeds for v in variants]
assert set(runs) == set(plan['suite']['runs'])
assert all(queue_start['jobs'][kind+'/'+run]['status'] == 'completed' for run in runs for kind in ['train', 'val'])

# Hash shared locked inputs once; verify that every run uses the same inputs.
reference = plan['signatures'][runs[0]]
for run in runs:
    signature = plan['signatures'][run]
    assert object_hash(no_digest(signature)) == signature['digest']
    for field in ['code', 'data', 'assets', 'init_sha256', 'environment']:
        assert signature[field] == reference[field], (run, field)
checked_hashes = 0
for field in ['code', 'data', 'assets']:
    for name, digest in reference[field].items():
        path = ROOT/name if field == 'code' else Path(name)
        assert sha(path) == digest, str(path)
        checked_hashes += 1
assert sha(reference['config']['RESEARCH']['BASE_WEIGHTS']) == reference['init_sha256']
print('VERIFIED locked code, Train/Val files, assets, and foundation weights', checked_hashes+1, flush=True)

prompts = ROOT/'data/Kvasir/Prompts_Folder'
train_records = read_text(prompts/'Train_text.xlsx')
val_records = read_text(prompts/'Val_text.xlsx')
assert len(train_records) == 800 and len(val_records) == 100
expected_pairs = {(r['Ground Truth'], r['Image']) for r in val_records}
assert len(expected_pairs) == 100
areas = []
for row in train_records:
    gt = cv2.imread(str(ROOT/'data/Kvasir/Train_Folder/label'/row['Ground Truth']), 0)
    assert gt is not None
    area = float((gt > 127).mean())
    if area:
        areas.append(area)
threshold = float(np.quantile(areas, .25))
summaries, cases, checks = {}, {}, {}
for run in runs:
    folder = OUT/'evaluations'/run/'Val/Kvasir'
    s, m = read(folder/'summary.json'), read(folder/'manifest.json')
    assert m['complete'] and m['n'] == 100 and s['n'] == 100 and s['n_small'] == 25
    assert s['mc_samples'] == 30 and s['split'] == 'Val' and s['run_id'] == run
    assert s['source_train_area_threshold'] == threshold
    assert not s['diagnostic'] and not s['subset_check']
    assert m['signature']['training_signature'] == plan['signatures'][run]['digest']
    assert m['signature']['checkpoint_sha256'] == s['checkpoint_sha256']
    assert object_hash(m['signature']) == m['digest']
    assert m['signature']['prompt_sha256'] == sha(prompts/'Val_text.xlsx')
    assert all(reference['data'][name] == digest for name, digest in m['signature']['evaluation_files'].items())
    assert sha(folder/'cases.csv') == m['cases_sha256']
    rows = numeric_rows(folder/'cases.csv')
    assert len(rows) == 100 and {(r['name'], r['image_name']) for r in rows} == expected_pairs
    recalculated = []
    for row in rows:
        with np.load(folder/row['probability_file']) as z:
            for key in ['mean', 'variance', 'entropy']:
                assert z[key].shape == (224, 224) and np.isfinite(z[key]).all(), (run, row['name'], key)
            mean = z['mean'].copy()
            assert mean.min() >= 0 and mean.max() <= 1 and z['variance'].min() >= 0
            assert z['entropy'].min() >= -1e-6 and z['entropy'].max() <= np.log(2)+1e-6
        mask = cv2.imread(str(folder/row['mask_file']), 0)
        assert mask is not None and np.array_equal(mask, (mean > .5).astype(np.uint8)*255)
        metrics = case_metrics(ROOT/'data/Kvasir/Val_Folder/label'/row['name'], mean)
        equal_summary(metrics, row)
        recalculated.append(dict(name=row['name'], **metrics))
    equal_summary(summarize(recalculated, threshold), s)
    summaries[run], cases[run] = s, {r['name']: r for r in recalculated}
    checks[run] = dict(recomputed_predictions=100, evaluation_manifest_verified=True)
    print('VERIFIED 100 saved probability arrays, PNGs, and metrics:', run, flush=True)

initial = {}
for run in runs:
    seed = plan['signatures'][run]['config']['seed']
    folder = OUT/run/'Kvasir/trained_models'/f'seed{seed}'
    p = folder/'MedCLIPSeg_unimedclip_ViT-B-16_latest.pth'
    assert sha(p) == summaries[run]['checkpoint_sha256']
    checkpoint = torch.load(p, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 99 and checkpoint['progress']['phase'] == 'epoch_complete'
    assert checkpoint['scheduler']['last_epoch'] == 100
    assert {int(x['step']) for x in checkpoint['optimizer']['state'].values()} == {3400}
    assert checkpoint['research_signature'] == plan['signatures'][run] == read(folder/'research_manifest.json')
    assert checkpoint['runtime'] == dict(num_workers=0, batch_size=24, train_size=800, val_size=100, gradient_checkpointing=True)
    assert all(k in checkpoint for k in ['optimizer', 'scheduler', 'rng_state', 'progress', 'runtime'])
    model = checkpoint['model']
    assert all(not v.is_floating_point() or torch.isfinite(v).all().item() for v in model.values())
    assert all(not isinstance(v, torch.Tensor) or not v.is_floating_point() or torch.isfinite(v).all().item()
               for state in checkpoint['optimizer']['state'].values() for v in state.values())
    assert sum(v.numel() for v in model.values()) == summaries[run]['total_parameters'] == 214580757
    trainable_count = sum(v.numel() for k, v in model.items() if k.startswith(('pvl_adapters.', 'mask_head.', 'upscale.')))
    assert trainable_count == summaries[run]['trainable_parameters'] == 18677268
    initial[run] = checkpoint['initial_fingerprints']
    assert fingerprint(model, False) == initial[run]['frozen'], ('Frozen weights changed', run)
    trained_fingerprint = fingerprint(model, True)
    assert trained_fingerprint != initial[run]['trainable']
    history = re.findall(r'EPOCH: (\d+) \| Training Loss: ([\d.eE+-]+) \| Validation Loss: ([\d.eE+-]+)', (folder/'log.txt').read_text(encoding='utf-8'))
    epochs = {int(e): [float(t), float(v)] for e, t, v in history}
    assert set(epochs) == set(range(1, 101)), (run, len(epochs))
    assert np.isfinite(list(epochs.values())).all()
    checks[run].update(epoch=100, optimizer_steps=3400, scheduler_steps=100,
                      checkpoint_sha256=summaries[run]['checkpoint_sha256'], frozen_parameters_unchanged=True,
                      trainable_parameters_updated=True, final_trainable_fingerprint=trained_fingerprint,
                      model_optimizer_finite=True, checkpoint_state_complete=True, logged_epochs=len(epochs),
                      train_loss_first10=float(np.mean([epochs[e][0] for e in range(1,11)])),
                      train_loss_last10=float(np.mean([epochs[e][0] for e in range(91,101)])),
                      final_training_loss=epochs[100][0], active_training_seconds=queue_start['jobs']['train/'+run]['active_seconds'])
    del model, checkpoint
    gc.collect()
    print('VERIFIED epoch100, Adam3400, frozen encoders, finite weights:', run, flush=True)
for seed in seeds:
    assert len({json.dumps(initial[f'{v}_seed{seed}'], sort_keys=True) for v in variants}) == 1
assert len({initial[f'paired_repeat_seed{s}']['trainable'] for s in seeds}) == 3
assert len({v['frozen'] for v in initial.values()}) == 1

keys = ['DSC_percent', 'small_DSC_percent', 'NSD_percent', 'micro_foreground_recall_percent',
        'micro_foreground_precision_percent', 'empty_positive', 'zero_overlap_positive',
        'positive_DSC_below_50_percent', 'positive_DSC_below_10_percent', 'brier_224']
aggregates = {v: {k: dict(mean=float(np.mean([summaries[f'{v}_seed{s}'][k] for s in seeds])),
                        sd=float(np.std([summaries[f'{v}_seed{s}'][k] for s in seeds], ddof=1))) for k in keys} for v in variants}

def distribution(rows):
    values = np.array([r['delta_pp'] for r in rows])
    ordered = sorted(rows, key=lambda r: r['delta_pp'], reverse=True)
    return dict(n=len(rows), mean_pp=float(values.mean()), median_pp=float(np.median(values)),
                improved=int((values>1e-8).sum()), declined=int((values< -1e-8).sum()), tied=int((abs(values)<=1e-8).sum()),
                top_two_contribution_pp=float(sum(r['delta_pp'] for r in ordered[:2])/len(rows)),
                remaining_mean_pp=float(np.mean([r['delta_pp'] for r in ordered[2:]])),
                best=ordered[:3], worst=ordered[-3:])

comparisons, paired_rows = {}, {}
for a, b in [('paired_image', 'paired_repeat'), ('paired_aug', 'paired_repeat'), ('paired_aug', 'paired_image')]:
    name = a+'_vs_'+b
    per_seed = {}
    for seed in seeds:
        ra, rb = f'{a}_seed{seed}', f'{b}_seed{seed}'
        rows = [dict(name=n, small=cases[ra][n]['gt_fraction']<=threshold,
                     delta_pp=100*(cases[ra][n]['DSC']-cases[rb][n]['DSC'])) for n in cases[ra]]
        per_seed[str(seed)] = dict(delta={k:summaries[ra][k]-summaries[rb][k] for k in keys},
                                   all_cases=distribution(rows), small_cases=distribution([r for r in rows if r['small']]))
    mean_rows = [dict(name=n, small=cases[f'{a}_seed666'][n]['gt_fraction']<=threshold,
                      delta_pp=float(np.mean([100*(cases[f'{a}_seed{s}'][n]['DSC']-cases[f'{b}_seed{s}'][n]['DSC']) for s in seeds])))
                 for n in cases[f'{a}_seed666']]
    comparisons[name] = dict(mean_delta={k:aggregates[a][k]['mean']-aggregates[b][k]['mean'] for k in keys},
                            per_seed=per_seed, seed_averaged_cases=distribution(mean_rows),
                            seed_averaged_small_cases=distribution([r for r in mean_rows if r['small']]))
    paired_rows[name] = mean_rows

# Completed Test jobs only: checksum and CSV aggregate review, not a full prediction recomputation.
queue_end = read(CONTROL/'queue.json')
partial_tests = {}
for key, job in queue_end['jobs'].items():
    if not key.startswith('test/') or job['status'] != 'completed':
        continue
    _, run, target = key.split('/')
    folder = OUT/'evaluations'/run/'Test'/target
    s, m = read(folder/'summary.json'), read(folder/'manifest.json')
    assert m['complete'] and m['cases_sha256'] == sha(folder/'cases.csv')
    assert s['checkpoint_sha256'] == summaries[run]['checkpoint_sha256']
    assert m['signature']['training_signature'] == plan['signatures'][run]['digest']
    assert s['mc_samples'] == 30 and s['n'] == plan['test_cohorts'][target]['n']
    equal_summary(summarize(numeric_rows(folder/'cases.csv'), threshold), s)
    partial_tests[key] = s

hard_names = ['cju40w3hbkwpn08015rbs3wko.png', 'cju76o55nymqd0871h31sph9w.png', 'cju89z6pqpqfx0817mfv8ixjc.png']
result = dict(checked_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), passed=True,
              scope='9 final checkpoints; 900 Val probability/PNG predictions and metrics recomputed on CPU; no GPU inference',
              plan_digest=plan['digest'], checked_input_hashes=checked_hashes+1, source_train_area_threshold=threshold,
              summaries=summaries, checks=checks, initial_fingerprints=initial, aggregates=aggregates, comparisons=comparisons,
              hard_cases={n:{run:cases[run][n]['DSC']*100 for run in runs} for n in hard_names},
              queue_start=queue_start, queue_end=queue_end,
              partial_tests_review_scope='Completed jobs only; CSV hashes and aggregate metrics checked; no Test NPZ or PNG recomputation',
              partial_tests=partial_tests, active_training_hours=sum(c['active_training_seconds'] for c in checks.values())/3600,
              statistical_significance_established=False, negative_val_cases=0)
(HERE/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
(HERE/'paired_cases.json').write_text(json.dumps(paired_rows, ensure_ascii=False, indent=2), encoding='utf-8')
print('PASS: all nine formal training and Val runs verified. Test snapshot', len(partial_tests), '/ 45', flush=True)
print(json.dumps(dict(aggregates=aggregates, comparison_deltas={k:v['mean_delta'] for k,v in comparisons.items()},
                     active_training_hours=result['active_training_hours']), ensure_ascii=False, indent=2))
