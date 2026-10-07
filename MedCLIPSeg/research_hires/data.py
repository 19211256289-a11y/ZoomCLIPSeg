"""Same images, augmentation and random draws as the accepted dataset; only the training mask is 448 instead of 224."""
import os

import cv2
import numpy as np

from datasets.dataloader import correct_dims
from research.data import DualScaleDataset, DualScaleTransform
from .common import HIRES


class HiResMaskDataset(DualScaleDataset):
    def __getitem__(self, idx):
        image_name, mask_name, text = self.data_pairs[idx]
        raw = cv2.imread(os.path.join(self.input_path, image_name))
        mask = cv2.imread(os.path.join(self.output_path, mask_name), 0)
        if raw is None or mask is None:
            raise ValueError(f'Unreadable image/mask: {self.task_name}/{image_name}')
        image = cv2.resize(raw, (self.image_size, self.image_size))
        detail = cv2.resize(raw, (self.detail_size, self.detail_size))
        # The only difference from the accepted dataset: the mask keeps the detail resolution.
        mask = cv2.resize(mask, (self.detail_size, self.detail_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask >= 127).astype(np.uint8)
        image, mask = correct_dims(image, mask)
        sample = dict(image=image, image_detail=detail, ground_truth_mask=mask, image_name=image_name,
                      mask_name=mask_name, text_prompt=text, dataset_name=self.task_name)
        return self.joint_transform(sample)


def make_dataset(cfg, split, records, condition=None):
    from research_detail_formal.data import PromptDataset
    if split not in ('Train', 'Val'):
        raise ValueError('This pilot does not access Test')
    inner = HiResMaskDataset(getattr(cfg.DATASET, split.upper()+'_PATH'), cfg.DATASET.NAME, records,
                             DualScaleTransform(split == 'Train'), image_size=224, detail_size=HIRES, upsampled=False)
    return PromptDataset(inner, generic=split == 'Val')
