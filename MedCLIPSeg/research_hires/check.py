"""Short GPU check before any pilot training: shapes, exact neutral initialisation, gradients, memory."""
import gc
from pathlib import Path

import monai
import torch
from torch.utils.data import DataLoader

from train_detail_formal import set_random_seed, forward_batch, calc_loss
from utils.main_utils import read_text
from .common import CONTROL, HIRES, make_config, now, write_json, object_hash, code_manifest
from .data import make_dataset
from .model import build_model


def run_check(suite):
    import research_detail_formal.data as accepted_data
    import research_detail_formal.objective as objective
    accepted_data.GENERIC = objective.GENERIC = suite['prompt']
    seed = suite['seeds'][0]
    configs = {arm: make_config(suite, f'{arm}_seed{seed}') for arm in ('sup448', 'detail448')}
    first = configs['detail448']
    records = read_text(Path(first.DATASET.TEXT_PROMPT_PATH)/'Train_text.xlsx')[:4]
    set_random_seed(seed)
    batch = next(iter(DataLoader(make_dataset(first, 'Train', records), batch_size=4, shuffle=False, num_workers=0)))
    assert tuple(batch['ground_truth_mask'].shape[-2:]) == (HIRES, HIRES), 'training mask must be 448'
    assert tuple(batch['image'].shape[-2:]) == (224, 224) and tuple(batch['image_detail'].shape[-2:]) == (HIRES, HIRES)
    outputs, report = {}, {}
    for arm, cfg in configs.items():
        set_random_seed(seed)
        model = build_model(cfg).to(cfg.MODEL.DEVICE)
        model.eval()
        set_random_seed(seed)
        with torch.no_grad():
            samples = forward_batch(model, batch, cfg.MODEL.DEVICE, 2)
        assert tuple(samples.shape) == (2, 4, HIRES, HIRES), f'unexpected output shape {tuple(samples.shape)}'
        outputs[arm] = samples.cpu()
        report[arm] = dict(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                           parameters=sum(p.numel() for p in model.parameters()))
        if arm == 'detail448':
            model.train()
            set_random_seed(seed)
            torch.cuda.reset_peak_memory_stats()
            ce = torch.nn.BCEWithLogitsLoss()
            dice = monai.losses.DiceLoss(include_background=True, sigmoid=True)
            loss, _ = objective.paired_objective(model, batch, cfg, ce, dice, forward_batch, calc_loss)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad), 'frozen parameter received a gradient'
            assert model.hires_out.weight.grad.abs().sum() > 0, '448 output layer received no gradient'
            assert model.detail_decoder.fuse224[0].weight.grad is not None, 'accepted detail decoder is not on the 448 path'
            report[arm].update(initial_paired_loss=float(loss.detach()), peak_reserved_gib_batch4=torch.cuda.max_memory_reserved()/2**30)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    assert torch.equal(outputs['sup448'], outputs['detail448']), 'zero-initialised 448 branch must not change the initial prediction'
    result = dict(passed=True, completed_at=now(), code_digest=object_hash(code_manifest()), arms=report,
                  neutral_initialisation_exact=True, scientific_training_started=False)
    write_json(CONTROL/'check.json', result)
    print('CHECK PASSED', report, flush=True)
    return result
