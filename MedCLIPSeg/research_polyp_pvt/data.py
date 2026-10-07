"""Images and masks as the released Polyp-PVT code prepares them: PIL bilinear resize to a square, ImageNet statistics.
Files are decoded with OpenCV, like in every accepted evaluation of this project.
Training images are decoded once and kept in memory (the released recipe uses no augmentation besides the three sizes,
which are produced on the GPU in the training loop)."""
import cv2
import numpy as np
import torch
from PIL import Image

from utils.main_utils import read_text
from .common import ROOT, sha256, object_hash

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def decode(path, flag=1):
    """Pixels as every accepted evaluation of this project decodes them (OpenCV)."""
    array = cv2.imread(str(path), flag)
    if array is None:
        raise ValueError(f'Unreadable image: {path}')
    return array


def image_tensor(path, size):
    """transforms.Resize((size, size)) + ToTensor + Normalize of the released loader."""
    image = Image.fromarray(cv2.cvtColor(decode(path), cv2.COLOR_BGR2RGB)).resize((size, size), Image.BILINEAR)
    array = (np.asarray(image, np.float32)/255-MEAN)/STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def mask_tensor(path, size):
    """Resize + ToTensor of the released loader. The mask is first binarised at 127, the definition of the ground truth
    in the metric code of this project; for binary mask files this changes nothing."""
    binary = np.where(decode(path, 0) > 127, 255, 0).astype(np.uint8)
    mask = Image.fromarray(binary).resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(mask, np.float32)[None]/255)


def train_records(protocol):
    folder = ROOT/'data'/protocol['dataset']
    listing = folder/'Prompts_Folder'/protocol['train_list']
    records = read_text(listing)
    if len(records) != protocol['train_n'] or len({r['Image'] for r in records}) != len(records):
        raise ValueError(f'Unexpected training list: {len(records)} records')
    return folder/'Train_Folder', listing, records


def train_seal(protocol):
    folder, listing, records = train_records(protocol)
    files = {f'{kind}/{r[key]}': sha256(folder/kind/r[key]) for r in records for kind, key in (('img', 'Image'), ('label', 'Ground Truth'))}
    return dict(n=len(records), list_sha256=sha256(listing), files_digest=object_hash(files))


def load_train(protocol, progress=None):
    """-> images [N, 3, S, S] and masks [N, 1, S, S] as float32 CPU tensors."""
    folder, _, records = train_records(protocol)
    size = protocol['recipe']['input_size']
    images = torch.empty(len(records), 3, size, size)
    masks = torch.empty(len(records), 1, size, size)
    for index, record in enumerate(records):
        if progress and index % 100 == 0:
            progress(index, len(records))
        images[index] = image_tensor(folder/'img'/record['Image'], size)
        masks[index] = mask_tensor(folder/'label'/record['Ground Truth'], size)
    if not (float(masks.min()) >= 0 and float(masks.max()) == 1. and .02 < float(masks.mean()) < .6):
        raise ValueError('Unexpected training masks')
    return images, masks
