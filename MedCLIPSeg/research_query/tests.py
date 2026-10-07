"""Numerical module checks plus the actual training loop's pause/resume harness."""
from local_runtime import configure_local_models
configure_local_models()
import ast,copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from research_formal.tests import old,toy_batch
from train_query import set_random_seed,capture_rng_state
from .common import ROOT,MODES,GENERIC,load_suite,make_config
from .data import PromptDataset
from .objective import prompt_views
from .model import RegionQuery,QueryCLIP
from trainers.medclipseg_unimedclip import CustomCLIP


class ToyQuery(old.ToyModel):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        if mode!='baseline':self.region_query=RegionQuery(4,2,mode)
    def forward(self,image,text,num_samples=1):
        raw=super().forward(image,text,num_samples)
        logits=raw[0] if self.training else raw.squeeze(0)
        if self.mode!='baseline':
            decoded=image.repeat(1,4,1,1)
            query=torch.stack((self.weight,self.bias,self.weight,self.bias)).unsqueeze(0).expand(image.shape[0],-1)
            delta=self.region_query(query,decoded,logits.unsqueeze(1))
            logits=logits+(decoded*delta[:,:,None,None]).sum(1)
        return (logits,raw[1]) if self.training else logits.unsqueeze(0)


class QueryTests(unittest.TestCase):
    def test_modes_share_prompts_geometry_rng(self):
        states=[];views=[]
        for mode in MODES:
            cfg=make_config(load_suite(),mode+'_seed666');set_random_seed(666)
            a,b,_=prompt_views(toy_batch(),cfg)
            self.assertEqual(a['text_prompt'],[GENERIC]);self.assertEqual(b['text_prompt'],[GENERIC])
            states.append(capture_rng_state());views.append((a,b))
        for i in (1,2):old.assert_equal(states[0],states[i]);old.assert_equal(views[0],views[i])

    def test_neutral_logits_match_native_and_module_consumes_no_rng(self):
        m=QueryCLIP.__new__(QueryCLIP);torch.nn.Module.__init__(m)
        m.patch_size=2;m.im_size=8;m.query_mode='contrast_query'
        m.mask_head=torch.nn.Linear(4,4);m.upscale=torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        m.region_query=RegionQuery(4,2)
        image,text=torch.randn(2,17,4),torch.randn(2,4)
        rng=capture_rng_state();actual=m.compute_seg_logits(image,text,2,8,8)
        expected=CustomCLIP.compute_seg_logits(m,image,text,2,8,8)
        old.assert_equal(actual,expected);old.assert_equal(rng,capture_rng_state())

    def test_gradients_saturation_and_equal_capacity(self):
        models=[RegionQuery(4,2,mode) for mode in MODES[1:]]
        self.assertEqual(sum(p.numel() for p in models[0].parameters()),sum(p.numel() for p in models[1].parameters()))
        for m in models:
            q=torch.randn(2,4,requires_grad=True);x=torch.randn(2,4,3,3,requires_grad=True)
            z=torch.randn(2,1,3,3,requires_grad=True)
            opt=torch.optim.Adam(m.parameters(),lr=.01)
            for step in range(2):
                opt.zero_grad();m(q,x,z).sum().backward()
                self.assertIsNone(z.grad)
                self.assertGreater(float(m.fuse[-1].weight.grad.abs().sum()),0)
                if step:self.assertGreater(float(m.query[1].weight.grad.abs().sum()),0)
                opt.step()
            for value in (-1e6,0.,1e6):
                self.assertTrue(torch.isfinite(m(q,x,torch.full_like(z,value))).all())

    def test_global_control_ignores_spatial_guidance_contrast_uses_it(self):
        global_m=RegionQuery(4,2,'global_query')
        torch.nn.init.normal_(global_m.fuse[-1].weight)
        contrast=copy.deepcopy(global_m);contrast.mode='contrast_query'
        q,x,z=torch.randn(2,4),torch.randn(2,4,3,3),torch.randn(2,1,3,3)*4
        self.assertTrue(torch.equal(global_m(q,x,z),global_m(q,x,-z)))
        self.assertFalse(torch.equal(contrast(q,x,z),contrast(q,x,-z)))

    def test_actual_production_repeated_pause_resume_with_module(self):
        namespace=old.make_namespace();logger=namespace['logger_config']
        tree=ast.parse((ROOT/'train_query.py').read_text(encoding='utf-8'))
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))],type_ignores=[]),'train_query production','exec'),namespace)
        namespace['logger_config']=logger
        original=old.make_cfg
        def dataset(cfg,split,records):
            return PromptDataset(old.ToyDataset(split.lower(),'toy',[],old.ToyTransform(augment=split=='Train')),split=='Val')
        for mode in MODES:
            def config():
                cfg=original();cfg.MODEL.VARIANT='paired_image';cfg.TRAIN.DICE_MODE='image'
                cfg.TRAIN.NEW_LEARNING_RATE=.003;cfg.TRAIN.MIN_LEARNING_RATE=.0001
                cfg.RESEARCH=SimpleNamespace(BASE_WEIGHTS='toy',RUN_ID=mode+'_seed666',PROMPT_MODE='generic',QUERY_MODE=mode,GENERIC_PROBABILITY=1.)
                return cfg
            with self.subTest(mode=mode),patch.object(old,'make_cfg',config), \
                 patch('research_query.model.build_model',lambda cfg:ToyQuery(cfg.RESEARCH.QUERY_MODE)), \
                 patch('research_query.common.make_signature',lambda cfg:{'digest':'query-toy'}), \
                 patch('research_query.data.make_dataset',dataset),tempfile.TemporaryDirectory() as directory:
                old.test_repeated_resume(namespace,Path(directory))

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main(verbosity=2)
