"""CLIPSeg with one fixed sentence: model, training loop (decoder only, pause/resume at epoch boundaries) and loading."""
import math
import os
import time

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from research_polyp_pvt.data import load_train
from .common import *


def seed_everything(value):
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


class FixedSentenceCLIPSeg(torch.nn.Module):
    """Released CLIPSeg with frozen CLIP encoders; every image receives the same sentence."""

    def __init__(self, protocol):
        super().__init__()
        from transformers import CLIPSegForImageSegmentation, CLIPTokenizer
        folder = str(model_dir(protocol))
        self.net = CLIPSegForImageSegmentation.from_pretrained(folder, local_files_only=True)
        tokens = CLIPTokenizer.from_pretrained(folder, local_files_only=True)(
            [protocol['model']['sentence']], padding='max_length', max_length=77, truncation=True, return_tensors='pt')
        self.register_buffer('input_ids', tokens['input_ids'], persistent=False)
        self.register_buffer('attention_mask', tokens['attention_mask'], persistent=False)
        for parameter in self.net.clip.parameters():
            parameter.requires_grad_(False)
        self.size = protocol['recipe']['input_size']
        import inspect
        self.extra = dict(interpolate_pos_encoding=True) if 'interpolate_pos_encoding' in inspect.signature(self.net.forward).parameters else {}

    def trainable(self):
        return [p for p in self.net.decoder.parameters() if p.requires_grad]

    def train(self, mode=True):
        super().train(mode)
        self.net.clip.eval()          # the frozen encoders never leave evaluation mode
        return self

    def forward(self, images):
        """images [B, 3, S, S] (ImageNet statistics) -> logits [B, 1, S, S]"""
        batch = images.shape[0]
        output = self.net(pixel_values=images, input_ids=self.input_ids.expand(batch, -1),
                          attention_mask=self.attention_mask.expand(batch, -1), **self.extra)
        logits = output.logits.reshape(batch, 1, *output.logits.shape[-2:])
        if logits.shape[-2:] != images.shape[-2:]:
            raise RuntimeError(f'Unexpected output shape: {tuple(output.logits.shape)}')
        return logits


def build(protocol, seed):
    seed_everything(seed)
    return FixedSentenceCLIPSeg(protocol).cuda()


def segmentation_loss(logits, target):
    probability = torch.sigmoid(logits)
    intersection = (probability*target).sum(dim=(1, 2, 3))
    dice = 1-(2*intersection+1)/(probability.sum(dim=(1, 2, 3))+target.sum(dim=(1, 2, 3))+1)
    return .5*dice.mean()+.5*F.binary_cross_entropy_with_logits(logits, target)


def augment(images, masks, degrees, generator):
    """Reflection (each axis with probability 0.5) and rotation within +/-degrees (probability 0.5), per image."""
    out_images, out_masks = [], []
    for index in range(images.shape[0]):
        image, mask = images[index], masks[index]
        draws = torch.rand(4, generator=generator)
        if draws[0] < .5:
            image, mask = image.flip(-1), mask.flip(-1)
        if draws[1] < .5:
            image, mask = image.flip(-2), mask.flip(-2)
        if draws[2] < .5:
            angle = float((2*draws[3]-1)*degrees)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        out_images.append(image)
        out_masks.append(mask)
    return torch.stack(out_images), torch.stack(out_masks)


def learning_rate(recipe, step, total):
    low, high = recipe['final_learning_rate'], recipe['learning_rate']
    return low+(high-low)*.5*(1+math.cos(math.pi*step/max(1, total-1)))


def load_data(protocol, group, train_n):
    images, masks = load_train(group_protocol(protocol, group, train_n))
    return images, (masks > .5).float()


def train_step(model, optimizer, x, y):
    optimizer.zero_grad()
    loss = segmentation_loss(model(x), y)
    loss.backward()
    optimizer.step()
    value = float(loss.detach())
    if not math.isfinite(value):
        raise RuntimeError('Non-finite training loss')
    return value


def smoke_test(protocol, group, data):
    """A few optimisation steps on a throw-away model; nothing is kept."""
    recipe = protocol['recipe']
    torch.cuda.reset_peak_memory_stats()
    model = build(protocol, protocol['seeds'][0])
    optimizer = torch.optim.Adam(model.trainable(), recipe['learning_rate'])
    model.train()
    images, masks = data
    generator = torch.Generator().manual_seed(0)
    losses, start = [], time.perf_counter()
    for step in range(3):
        index = slice(step*recipe['batch_size'], (step+1)*recipe['batch_size'])
        x, y = augment(images[index].cuda(), masks[index].cuda(), recipe['rotation_degrees'], generator)
        losses.append(train_step(model, optimizer, x, y))
    torch.cuda.synchronize()
    seconds = (time.perf_counter()-start)/3
    model.eval()
    with torch.no_grad():
        single = model(images[:1].cuda())
    if tuple(single.shape) != (1, 1, recipe['input_size'], recipe['input_size']):
        raise RuntimeError(f'Unexpected output shape: {tuple(single.shape)}')
    result = dict(parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.trainable()),
                  losses_first_steps=losses, seconds_per_step_incl_warmup=seconds,
                  peak_gpu_memory_gb=torch.cuda.max_memory_allocated()/2**30,
                  gpu_total_memory_gb=torch.cuda.get_device_properties(0).total_memory/2**30,
                  device=torch.cuda.get_device_name(), torch=torch.__version__)
    del model, optimizer, single
    torch.cuda.empty_cache()
    return result


def save_atomic(obj, path):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, path)


def train(protocol, group, seed, plan_digest, control, data):
    """Trains the decoder and writes final.pth (decoder weights of the last epoch)."""
    recipe, folder = protocol['recipe'], training_dir(group, seed)
    final, resume = folder/'final.pth', folder/'resume.pth'
    manifest = maybe(folder/'manifest.json')
    if manifest.get('complete') and manifest['plan_digest'] == plan_digest and final.is_file() and sha256(final) == manifest['sha256']:
        return
    folder.mkdir(parents=True, exist_ok=True)
    images, masks = data
    model = build(protocol, seed)
    optimizer = torch.optim.Adam(model.trainable(), recipe['learning_rate'])
    first, history = 1, []
    if resume.is_file():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['plan_digest'] == plan_digest and state['seed'] == seed:
            model.net.decoder.load_state_dict(state['decoder'])
            optimizer.load_state_dict(state['optimizer'])
            first, history = state['epoch']+1, state['history']
            print(f'seed {seed}: continuing after epoch {state["epoch"]}', flush=True)
    n, batch = images.shape[0], recipe['batch_size']
    steps = math.ceil(n/batch)
    total = steps*recipe['epochs']
    torch.backends.cudnn.benchmark = True
    for epoch in range(first, recipe['epochs']+1):
        seed_everything(seed*1000+epoch)        # a continued run uses the same random numbers as an uninterrupted one
        generator = torch.Generator().manual_seed(seed*1000+epoch)
        order = torch.randperm(n, generator=generator)
        model.train()
        running, start = 0., time.perf_counter()
        for step in range(steps):
            control.check(seed=seed, epoch=epoch)
            for group_ in optimizer.param_groups:
                group_['lr'] = learning_rate(recipe, (epoch-1)*steps+step, total)
            index = order[step*batch:(step+1)*batch]
            x, y = augment(images[index].cuda(), masks[index].cuda(), recipe['rotation_degrees'], generator)
            running += train_step(model, optimizer, x, y)
            if step % 5 == 0:
                control.update('running', phase='training', seed=seed, epoch=epoch, total_epochs=recipe['epochs'],
                               step=step+1, total_steps=steps, loss=running/(step+1))
        seconds = time.perf_counter()-start
        history.append(dict(epoch=epoch, loss=running/steps, seconds=seconds))
        if epoch < recipe['epochs']:
            save_atomic(dict(plan_digest=plan_digest, seed=seed, epoch=epoch, history=history,
                             decoder=model.net.decoder.state_dict(), optimizer=optimizer.state_dict()), resume)
        if epoch % 5 == 0 or epoch == 1:
            print(f'seed {seed} epoch {epoch}/{recipe["epochs"]} loss {running/steps:.4f} ({seconds:.0f} s)', flush=True)
    save_atomic(model.net.decoder.state_dict(), final)
    write_json(folder/'manifest.json', dict(complete=True, plan_digest=plan_digest, seed=seed, epochs=recipe['epochs'],
                                            sha256=sha256(final), history=history, finished_at=now(),
                                            active_seconds=sum(h['seconds'] for h in history),
                                            device=torch.cuda.get_device_name(), torch=torch.__version__))
    resume.unlink(missing_ok=True)   # own temporary file; the final weights are kept
    del model, optimizer
    torch.cuda.empty_cache()


def load_model(protocol, group, seed, plan_digest):
    folder = training_dir(group, seed)
    manifest = read(folder/'manifest.json')
    if not manifest.get('complete') or manifest['plan_digest'] != plan_digest or sha256(folder/'final.pth') != manifest['sha256']:
        raise RuntimeError(f'Training of {group}, seed {seed} is not complete or its weights changed')
    model = FixedSentenceCLIPSeg(protocol)
    model.net.decoder.load_state_dict(torch.load(folder/'final.pth', map_location='cpu'))
    return model.cuda().eval(), manifest['sha256']
