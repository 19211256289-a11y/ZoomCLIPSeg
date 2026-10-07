"""Preserve the BGR224 path. Validation always uses the fixed class sentence."""
from research.data import DualScaleDataset,DualScaleTransform
from .common import GENERIC

class PromptDataset:
    def __init__(self,inner,generic=False):self.inner,self.generic=inner,generic
    def __len__(self):return len(self.inner)
    def __getitem__(self,index):
        sample=self.inner[index]
        if self.generic:
            sample=dict(sample)
            sample['text_prompt']=GENERIC
        return sample

def make_dataset(cfg,split,records,condition=None):
    if split not in ('Train','Val'):raise ValueError('This pilot does not access Test')
    if condition not in (None,'generic'):raise ValueError(condition)
    inner=DualScaleDataset(getattr(cfg.DATASET,split.upper()+'_PATH'),cfg.DATASET.NAME,records,
        DualScaleTransform(split=='Train'),image_size=224,detail_size=448,upsampled=False)
    return PromptDataset(inner,generic=split=='Val')
