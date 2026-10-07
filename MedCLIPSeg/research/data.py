"""Two full-image resolutions with the original 224 preprocessing preserved."""
import os
import random

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F

from datasets.dataloader import DatasetSegmentation, CLIP_NORMALIZE, correct_dims


DETAIL_VARIANTS = {'detail', 'detail_beta', 'detail_upsampled', 'detail_scalar_beta'}


class DualScaleDataset(DatasetSegmentation):
    def __init__(self, *args, detail_size=448, upsampled=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.detail_size = detail_size
        self.upsampled = upsampled

    def __getitem__(self, idx):
        image_name, mask_name, text = self.data_pairs[idx]
        raw = cv2.imread(os.path.join(self.input_path, image_name))
        mask = cv2.imread(os.path.join(self.output_path, mask_name), 0)
        if raw is None or mask is None:
            raise ValueError(f'Unreadable image/mask: {self.task_name}/{image_name}')
        # Match the author's first resize exactly, including OpenCV BGR.
        image = cv2.resize(raw, (self.image_size, self.image_size))
        detail = cv2.resize(image if self.upsampled else raw,
                            (self.detail_size, self.detail_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask >= 127).astype(np.uint8)
        image, mask = correct_dims(image, mask)
        sample = dict(image=image, image_detail=detail, ground_truth_mask=mask,
                      image_name=image_name, mask_name=mask_name, text_prompt=text,
                      dataset_name=self.task_name)
        return self.joint_transform(sample)


class DualScaleTransform:
    def __init__(self, training=False):
        self.training = training

    def __call__(self, sample):
        image = Image.fromarray(sample['image'].astype(np.uint8))
        detail = Image.fromarray(sample['image_detail'].astype(np.uint8))
        mask_array = sample['ground_truth_mask']
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        mask = Image.fromarray(mask_array.astype(np.uint8))
        # Same number/order of Python random draws as RandomGenerator.
        if self.training and random.random() > .5:
            angle = random.randint(-20, 20)
            image, detail, mask = image.rotate(angle), detail.rotate(angle), mask.rotate(angle)
        sample['image'] = CLIP_NORMALIZE(F.to_tensor(image))
        sample['image_detail'] = CLIP_NORMALIZE(F.to_tensor(detail))
        import torch
        sample['ground_truth_mask'] = torch.from_numpy(np.array(mask, np.uint8)).long()
        return sample


def make_dataset(cfg, split, records):
    from datasets.dataloader import RandomGenerator, ValGenerator
    training = split == 'Train'
    path = getattr(cfg.DATASET, split.upper() + '_PATH')
    if cfg.MODEL.VARIANT in DETAIL_VARIANTS:
        return DualScaleDataset(path, cfg.DATASET.NAME, records, DualScaleTransform(training),
                                image_size=cfg.DATASET.SIZE, detail_size=cfg.DATASET.DETAIL_SIZE,
                                upsampled=cfg.MODEL.VARIANT == 'detail_upsampled')
    transform = RandomGenerator([cfg.DATASET.SIZE]*2) if training else ValGenerator([cfg.DATASET.SIZE]*2)
    return DatasetSegmentation(path, cfg.DATASET.NAME, records, transform, image_size=cfg.DATASET.SIZE)
