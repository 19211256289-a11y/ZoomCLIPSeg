"""Preserve the BGR224 path. Validation always uses the fixed class sentence."""
from research_formal.data import make_dataset as original_dataset
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
    if condition not in (None,'generic','original'):raise ValueError(condition)
    generic=(split=='Val' and condition!='original')
    return PromptDataset(original_dataset(cfg,split,records),generic)
