"""CPU correctness checks, including the real training pause state machine."""
from local_runtime import configure_local_models
configure_local_models()

import copy
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import monai
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from train import calc_loss, set_random_seed
from utils.main_utils import read_text
from datasets.dataloader import DatasetSegmentation, RandomGenerator, ValGenerator
from trainers.layers import ProbCrossAttention
from .common import ROOT, WORKSPACE, load_suite, make_config, assert_signature, object_hash
from .data import make_dataset
from .model import initialize_from_checkpoint, fusion_stage
from .modules import DetailEncoder, DetailDecoder, SpatialBeta, EvidenceAdapter, attention_with_visual_beta
from .metrics import summarize
from .report import gate


class ResearchTests(unittest.TestCase):
    def test_original_preprocessing_and_real_detail(self):
        suite = load_suite('pilot_kvasir')
        cfg = make_config(suite, 'detail')
        records = read_text(Path(cfg.DATASET.TEXT_PROMPT_PATH)/'Train_text.xlsx')[:3]
        original = DatasetSegmentation(cfg.DATASET.TRAIN_PATH, 'Kvasir', records, RandomGenerator([224,224]))
        dual = make_dataset(cfg, 'Train', records)
        cfg.MODEL.VARIANT = 'detail_upsampled'
        upsampled = make_dataset(cfg, 'Train', records)
        for index in range(len(records)):
            for seed in (1, 2):
                set_random_seed(seed)
                expected = original[index]
                set_random_seed(seed)
                actual = dual[index]
                self.assertTrue(torch.equal(expected['image'], actual['image']))
                self.assertTrue(torch.equal(expected['ground_truth_mask'], actual['ground_truth_mask']))
                self.assertEqual(actual['image_detail'].shape, (3,448,448))
                set_random_seed(seed)
                fake = upsampled[index]
                self.assertFalse(torch.equal(actual['image_detail'], fake['image_detail']))

    def test_dice_modes_and_empty_foreground(self):
        cfg = SimpleNamespace(TRAIN=SimpleNamespace(DICE_MODE='image', DICE_WEIGHT=1., CE_WEIGHT=0.))
        for batch in (1, 2):
            logits = torch.linspace(-3, 3, batch*64).reshape(batch,8,8)
            for empty in (True, False):
                target = torch.zeros_like(logits)
                if not empty:
                    target[:,2:4,3:5] = 1
                actual = calc_loss(logits, target, nn.BCEWithLogitsLoss(),
                                   monai.losses.DiceLoss(include_background=True, sigmoid=True), cfg)
                prob = logits.sigmoid()
                expected = (1-(2*(prob*target).sum((1,2))+1e-5)/(prob.sum((1,2))+target.sum((1,2))+1e-5)).mean()
                torch.testing.assert_close(actual, expected)
                cfg.TRAIN.DICE_MODE = 'legacy'
                old = monai.losses.DiceLoss(include_background=False, sigmoid=True)
                torch.testing.assert_close(calc_loss(logits,target,nn.BCEWithLogitsLoss(),old,cfg), old(logits,target))
                cfg.TRAIN.DICE_MODE = 'image'

    def test_neutral_attention_and_beta_bounds(self):
        attention = ProbCrossAttention(16)
        query, context = torch.randn(2,7,16), torch.randn(2,10,16)
        torch.manual_seed(15)
        reference = attention(query, context)
        torch.manual_seed(15)
        actual = attention_with_visual_beta(attention, query, context, torch.full((2,10),2.35))
        torch.testing.assert_close(actual, reference, atol=1e-7, rtol=1e-7)
        for scalar in (False,True):
            predictor = SpatialBeta(16, scalar=scalar)
            visual, detail = torch.randn(2,10,16), torch.randn(2,9,128)
            beta = predictor(visual, detail)
            torch.testing.assert_close(beta, torch.full((2,10),2.35))
            with torch.no_grad():
                if scalar: predictor.raw.fill_(100)
                else: predictor.predict[-1].bias.fill_(-100)
            beta = predictor(visual, detail)
            self.assertTrue(bool((beta >= 1.175-1e-6).all() and (beta <= 3.525+1e-6).all()))
            torch.testing.assert_close(beta[:,0], torch.full((2,),2.35))

    def test_residual_initialization_and_upstream_gradient(self):
        encoder, decoder = DetailEncoder(), DetailDecoder(16)
        optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=1e-3)
        image, global_features = torch.randn(1,3,448,448), torch.randn(1,16,56,56)
        for step in range(2):
            optimizer.zero_grad()
            output = decoder(global_features, encoder(image))
            if step == 0:
                self.assertEqual(output.count_nonzero().item(), 0)
            target = torch.zeros_like(output); target[:,20:45,30:55] = 1
            (output-target).square().mean().backward()
            if step == 1:
                self.assertGreater(encoder.stages[0][0][0].weight.grad.abs().sum().item(),0)
            optimizer.step()

    def test_checkpointing_preserves_detail_and_stochastic_gradients(self):
        class TextLayer(nn.Module):
            def __init__(self):
                super().__init__(); self.proj=nn.Linear(16,16)
            def forward(self,x,attention_mask=None):
                return (torch.tanh(self.proj(x)),)
        adapter = EvidenceAdapter(in_channels_vis=16,in_channels_txt=16,adapter_channels=16,beta=2.35,gate_init=0.)
        with torch.no_grad(): adapter.beta_predictor.predict[-1].weight.normal_(0,.05)
        block, layer = nn.Linear(16,16), TextLayer()
        modules = nn.ModuleList([block,layer,adapter])
        images,text,detail = torch.randn(10,2,16),torch.randn(2,7,16),torch.randn(2,9,128)
        results=[]
        for use_checkpoint in (False,True):
            modules.zero_grad(set_to_none=True)
            xi,xt,dt = [x.detach().clone().requires_grad_() for x in (images,text,detail)]
            args=(xi,xt,None,dt,block,layer,adapter)
            torch.manual_seed(81)
            out = checkpoint(fusion_stage,*args,use_reentrant=False,preserve_rng_state=True) if use_checkpoint else fusion_stage(*args)
            sum(x.square().mean() for x in out).backward()
            results.append(([x.detach().clone() for x in out], [p.grad.detach().clone() for p in modules.parameters()],dt.grad.clone(),torch.get_rng_state()))
        for a,b in zip(results[0][0]+results[0][1]+list(results[0][2:]),results[1][0]+results[1][1]+list(results[1][2:])):
            torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_loading_and_signature_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.pth'
            torch.save({'model':{'weight':torch.zeros(3,3)}},path)
            with self.assertRaises(ValueError): initialize_from_checkpoint(nn.Linear(2,2),path)
        with self.assertRaises(ValueError): assert_signature({'digest':'old'},{'digest':'new'})

    def test_metrics_and_screening_missing_foreground(self):
        rows=[dict(name='normal',gt_pixels=0,pred_pixels=0,tp_pixels=0,fp_pixels=0,fn_pixels=0,gt_fraction=0.,DSC=1.,NSD=1.,brier_224=0.)]
        result=summarize(rows,.01)
        self.assertIsNone(result['small_DSC_percent'])
        self.assertFalse(gate(result,result,2)['passed'])

    def test_research_pause_resume_real_engine(self):
        spec=importlib.util.spec_from_file_location('pause_reference',WORKSPACE/'reports/test_training_pause.py')
        old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        original_cfg=old.make_cfg
        class ToyDetail(old.ToyModel):
            def __init__(self):
                super().__init__(); self.detail_encoder=nn.Linear(1,1)
            def forward(self,image,text,num_samples=1,image_detail=None):
                base=super().forward(image,text,num_samples)
                extra=self.detail_encoder(image_detail[:,0].unsqueeze(-1)).squeeze(-1)
                self.last_regularization=self.detail_encoder.weight.square().mean()*.01
                if self.training: return base[0]+extra,base[1]
                return base+extra.unsqueeze(0)
        class ToyDataset(old.ToyDataset):
            def __getitem__(self,index):
                batch=super().__getitem__(index); batch['image_detail']=batch['image'].clone(); return batch
        def make_cfg():
            cfg=original_cfg(); cfg.MODEL.VARIANT='detail_beta'
            cfg.TRAIN.DICE_MODE='image'; cfg.TRAIN.NEW_LEARNING_RATE=.005
            cfg.TRAIN.BETA_WEIGHT=.001; cfg.TRAIN.MIN_LEARNING_RATE=.0001
            cfg.RESEARCH=SimpleNamespace(INIT_FROM='toy-init')
            return cfg
        def dataset(cfg,split,records):
            return ToyDataset(split.lower(),'toy',[],old.ToyTransform(augment=split=='Train'))
        namespace=old.make_namespace()
        with patch.object(old,'make_cfg',make_cfg), patch('research.model.build_model',lambda cfg:ToyDetail()), \
             patch('research.model.initialize_from_checkpoint',lambda *args:[]), \
             patch('research.common.make_signature',lambda cfg:{'digest':'toy-stable'}), \
             patch('research.data.make_dataset',dataset), tempfile.TemporaryDirectory() as directory:
            old.test_repeated_resume(namespace,Path(directory))


def main():
    torch.set_num_threads(2)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ResearchTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__=='__main__': main()
