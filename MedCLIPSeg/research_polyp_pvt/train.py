"""Training loop of the released Polyp-PVT script (Train.py), with pause/resume at epoch boundaries.

Kept from the released script: AdamW, constant learning rate, element-wise gradient clipping, three input sizes per
mini-batch (bilinear, align_corners=True), structure loss on both outputs, no further augmentation.
Different (listed in the protocol): seeded mini-batch order, last-epoch weights instead of test-selected weights."""
import math
import os
import time

import torch
import torch.nn.functional as F

from .common import *
from .lib.pvt import PolypPVT


def structure_loss(pred, mask):
    weit = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    # The released code calls binary_cross_entropy_with_logits(pred, mask, reduce='none'). PyTorch reads the non-empty
    # string as the legacy flag reduce=True, i.e. the mean over all elements; this is what was executed and is kept.
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit*wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2, 3))
    union = ((pred+mask)*weit).sum(dim=(2, 3))
    wiou = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


def clip_gradient(optimizer, limit):
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if parameter.grad is not None:
                parameter.grad.data.clamp_(-limit, limit)


def seed_everything(value):
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def build(seed):
    """Fresh model: ImageNet backbone, randomly initialised decoder (drawn from the seed)."""
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f'Backbone weights not found: {WEIGHTS}')
    seed_everything(seed)
    model = PolypPVT(backbone_weights=WEIGHTS)
    if model.backbone_keys_loaded < .9*model.backbone_keys_total:
        raise RuntimeError(f'Only {model.backbone_keys_loaded} of {model.backbone_keys_total} backbone tensors were found in {WEIGHTS}')
    return model.cuda()


def make_optimizer(model, recipe):
    return torch.optim.AdamW(model.parameters(), recipe['learning_rate'], weight_decay=recipe['weight_decay'])


def train_step(model, optimizer, x, y, recipe):
    """One mini-batch at the three sizes; returns the loss at the native size."""
    native = None
    for size in SIZES:
        optimizer.zero_grad()
        if size != recipe['input_size']:
            images = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=True)
            targets = F.interpolate(y, size=(size, size), mode='bilinear', align_corners=True)
        else:
            images, targets = x, y
        p1, p2 = model(images)
        loss = structure_loss(p1, targets)+structure_loss(p2, targets)
        loss.backward()
        clip_gradient(optimizer, recipe['gradient_clip'])
        optimizer.step()
        if size == recipe['input_size']:
            native = float(loss.detach())
    if native is None or not math.isfinite(native):
        raise RuntimeError('Non-finite training loss')
    return native


def smoke_test(protocol, data):
    """A few real optimisation steps on a throw-away model: surfaces missing files, shape errors and memory limits
    within a minute. Nothing from this model is kept."""
    recipe = protocol['recipe']
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()
    model = build(protocol['seeds'][0])
    optimizer = make_optimizer(model, recipe)
    model.train()
    images, masks = data
    losses, start = [], time.perf_counter()
    for step in range(3):
        index = slice(step*recipe['batch_size'], (step+1)*recipe['batch_size'])
        losses.append(train_step(model, optimizer, images[index].cuda(), masks[index].cuda(), recipe))
    torch.cuda.synchronize()
    seconds = (time.perf_counter()-start)/3
    model.eval()
    with torch.no_grad():
        p1, p2 = model(images[:1].cuda())
    if tuple(p1.shape) != (1, 1, recipe['input_size'], recipe['input_size']) or p1.shape != p2.shape:
        raise RuntimeError(f'Unexpected output shape: {tuple(p1.shape)}')
    result = dict(backbone_tensors_loaded=model.backbone_keys_loaded, backbone_tensors_total=model.backbone_keys_total,
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  losses_first_steps=losses, seconds_per_step_incl_warmup=seconds,
                  peak_gpu_memory_gb=torch.cuda.max_memory_allocated()/2**30,
                  gpu_total_memory_gb=torch.cuda.get_device_properties(0).total_memory/2**30,
                  device=torch.cuda.get_device_name(), torch=torch.__version__)
    del model, optimizer, p1, p2
    torch.cuda.empty_cache()
    return result


def save_atomic(obj, path):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, path)


def train_seed(protocol, seed, plan_digest, control, data):
    """Trains one model and writes final.pth (last epoch). A paused run continues from the last completed epoch."""
    recipe, folder = protocol['recipe'], training_dir(seed)
    final, resume = folder/'final.pth', folder/'resume.pth'
    manifest = maybe(folder/'manifest.json')
    if manifest.get('complete') and manifest['plan_digest'] == plan_digest and final.is_file() and sha256(final) == manifest['sha256']:
        return
    folder.mkdir(parents=True, exist_ok=True)
    images, masks = data
    model = build(seed)
    optimizer = make_optimizer(model, recipe)
    first, history = 1, []
    if resume.is_file():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['plan_digest'] == plan_digest and state['seed'] == seed:
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            first, history = state['epoch']+1, state['history']
            print(f'seed {seed}: continuing after epoch {state["epoch"]}', flush=True)
    n, batch = images.shape[0], recipe['batch_size']
    steps = math.ceil(n/batch)
    torch.backends.cudnn.benchmark = True
    for epoch in range(first, recipe['epochs']+1):
        # Everything random in an epoch (mini-batch order, stochastic depth) is drawn from (seed, epoch), so that a run
        # continued after a pause uses the same random numbers as an uninterrupted run.
        seed_everything(seed*1000+epoch)
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed*1000+epoch))
        model.train()
        total, start = 0., time.perf_counter()
        for step in range(steps):
            control.check(seed=seed, epoch=epoch)
            index = order[step*batch:(step+1)*batch]
            total += train_step(model, optimizer, images[index].cuda(), masks[index].cuda(), recipe)
            if step % 5 == 0:
                control.update('running', phase='training', seed=seed, epoch=epoch, total_epochs=recipe['epochs'],
                               step=step+1, total_steps=steps, loss=total/(step+1))
        seconds = time.perf_counter()-start
        history.append(dict(epoch=epoch, loss=total/steps, seconds=seconds))
        if epoch < recipe['epochs']:
            save_atomic(dict(plan_digest=plan_digest, seed=seed, epoch=epoch, history=history,
                             model=model.state_dict(), optimizer=optimizer.state_dict()), resume)
        print(f'seed {seed} epoch {epoch}/{recipe["epochs"]} loss {total/steps:.4f} ({seconds:.0f} s)', flush=True)
    save_atomic(model.state_dict(), final)
    write_json(folder/'manifest.json', dict(complete=True, plan_digest=plan_digest, seed=seed, epochs=recipe['epochs'],
                                            sha256=sha256(final), history=history, finished_at=now(),
                                            active_seconds=sum(h['seconds'] for h in history),
                                            device=torch.cuda.get_device_name(), torch=torch.__version__))
    resume.unlink(missing_ok=True)   # own temporary file; the final weights are kept
    del model, optimizer
    torch.cuda.empty_cache()
