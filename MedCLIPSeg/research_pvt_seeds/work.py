"""Training and model loading for one group and seed. The accepted Polyp-PVT training loop and inference are used
unchanged; only the folders and the training list differ."""
import contextlib

import torch

import research_polyp_pvt.train as accepted_train
from research_polyp_pvt.data import load_train, train_records
from research_polyp_pvt.lib.pvt import PolypPVT
from .common import WEIGHTS, read, maybe, sha256, training_dir, group_protocol


@contextlib.contextmanager
def output_folder(folder):
    """The accepted training function writes to training_dir(seed) of its own module; point that name at our folder."""
    original = accepted_train.training_dir
    accepted_train.training_dir = lambda seed: folder
    try:
        yield
    finally:
        accepted_train.training_dir = original


def count_training_images(protocol, group):
    spec = protocol['groups'][group]
    from utils.main_utils import read_text
    from .common import ROOT
    return len(read_text(ROOT/'data'/spec['dataset']/'Prompts_Folder'/spec['train_list']))


def load_data(protocol, group, train_n):
    return load_train(group_protocol(protocol, group, train_n))


def train(protocol, group, seed, plan_digest, control, data, train_n):
    with output_folder(training_dir(group, seed)):
        accepted_train.train_seed(group_protocol(protocol, group, train_n), seed, plan_digest, control, data)


def smoke_test(protocol, group, data, train_n):
    return accepted_train.smoke_test(group_protocol(protocol, group, train_n), data)


def load_model(group, seed, plan_digest):
    folder = training_dir(group, seed)
    manifest = read(folder/'manifest.json')
    if not manifest.get('complete') or manifest['plan_digest'] != plan_digest or sha256(folder/'final.pth') != manifest['sha256']:
        raise RuntimeError(f'Training of {group}, seed {seed} is not complete or its weights changed')
    model = PolypPVT(backbone_weights=WEIGHTS)
    model.load_state_dict(torch.load(folder/'final.pth', map_location='cpu'))
    return model.cuda().eval(), manifest['sha256']
