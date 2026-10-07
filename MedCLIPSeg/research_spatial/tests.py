"""Spatial routing, exact neutral logits, gradients, checkpointing and real resume."""
from local_runtime import configure_local_models
configure_local_models()
import ast,copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.utils.checkpoint import checkpoint
from research_formal.tests import old,toy_batch
from train_spatial import set_random_seed,capture_rng_state
from .common import ROOT,MODES,GENERIC,load_suite,make_config
from .data import PromptDataset
from .objective import prompt_views
from .model import SpatialQuery,SpatialCLIP
from trainers.medclipseg_unimedclip import CustomCLIP


class ToySpatial(old.ToyModel):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        if mode!='baseline':self.spatial_query=SpatialQuery(4,4,4,4,mode)
    def forward(self,image,text,num_samples=1):
        raw=super().forward(image,text,num_samples)
        logits=raw[0] if self.training else raw.squeeze(0)
        if self.mode!='baseline':
            query=torch.stack((self.weight,self.bias,self.weight,self.bias)).unsqueeze(0).expand(image.shape[0],-1)
            logits=logits+self.spatial_query(query,image.repeat(1,4,1,1)).squeeze(1)
        return (logits,raw[1]) if self.training else logits.unsqueeze(0)


class SpatialTests(unittest.TestCase):
    def test_problem_specific_gate_and_outlier_protection(self):
        from .gate import screening_gate
        keys=dict(DSC_percent=87.,small_DSC_percent=84.,micro_foreground_precision_percent=89.,
                  micro_foreground_recall_percent=84.,zero_overlap_positive=0,source_train_area_threshold=.06)
        summaries={mode:dict(keys) for mode in MODES}
        summaries['spatial_mix'].update(DSC_percent=86.8,small_DSC_percent=84.7)
        source=[dict(name=str(i),gt_fraction=.03,DSC=.80) for i in range(4)]
        cases={mode:copy.deepcopy(source) for mode in MODES}
        for r,v in zip(cases['spatial_mix'],(.82,.82,.794,.794)):r['DSC']=v
        g=load_suite()['screening_gate']
        self.assertTrue(screening_gate(summaries,cases,g)['passed'])
        # An isolated gain must not hide declines in most small lesions.
        for r,v in zip(cases['spatial_mix'],(.99,.79,.79,.79)):r['DSC']=v
        self.assertFalse(screening_gate(summaries,cases,g)['passed'])
        for r in cases['spatial_mix']:r['DSC']=.81
        summaries['spatial_mix']['DSC_percent']=86.0
        self.assertFalse(screening_gate(summaries,cases,g)['passed'])

    def test_modes_share_data_and_rng(self):
        states=[];views=[]
        for mode in MODES:
            cfg=make_config(load_suite(),mode+'_seed666');set_random_seed(666)
            a,b,_=prompt_views(toy_batch(),cfg)
            self.assertEqual(a['text_prompt'],[GENERIC]);self.assertEqual(b['text_prompt'],[GENERIC])
            states.append(capture_rng_state());views.append((a,b))
        for i in (1,2):old.assert_equal(states[0],states[i]);old.assert_equal(views[0],views[i])

    def test_neutral_output_and_no_extra_rng(self):
        m=SpatialCLIP.__new__(SpatialCLIP);torch.nn.Module.__init__(m)
        m.patch_size=2;m.im_size=8;m.query_mode='spatial_mix'
        m.mask_head=torch.nn.Linear(4,4);m.upscale=torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        m.spatial_query=SpatialQuery(4,4,4)
        image,text=torch.randn(2,17,4),torch.randn(2,4)
        rng=capture_rng_state();actual=m.compute_seg_logits(image,text,2,8,8)
        old.assert_equal(actual,CustomCLIP.compute_seg_logits(m,image,text,2,8,8))
        old.assert_equal(rng,capture_rng_state())

    def test_expert_and_router_learning_after_neutral_start(self):
        for mode in MODES[1:]:
            torch.manual_seed(32);m=SpatialQuery(4,4,4,4,mode)
            q,x=torch.randn(2,4),torch.randn(2,4,5,5);opt=torch.optim.Adam(m.parameters(),lr=.01)
            target=torch.randn(2,1,5,5)
            for step in range(2):
                opt.zero_grad();loss=(m(q,x)-target).square().mean();loss.backward()
                self.assertGreater(float(m.experts[-1].weight.grad.abs().sum()),0)
                if step:
                    self.assertGreater(float(m.experts[1].weight.grad.abs().sum()),0)
                    self.assertGreater(float(m.router[-1].weight.grad.abs().sum()),0)
                else:self.assertEqual(float(m.router[-1].weight.grad.abs().sum()),0)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))
                opt.step()
            with torch.no_grad():
                expert_weights=m.experts[-1].weight.reshape(4,4,4)
                self.assertFalse(torch.equal(expert_weights[0],expert_weights[1]))

    def test_equal_capacity_and_explicit_mixture_formula(self):
        torch.manual_seed(33);spatial=SpatialQuery(4,4,4)
        torch.nn.init.normal_(spatial.experts[-1].weight)
        global_m=copy.deepcopy(spatial);global_m.mode='global_mix'
        self.assertEqual(sum(p.numel() for p in spatial.parameters()),sum(p.numel() for p in global_m.parameters()))
        q,x=torch.randn(2,4),torch.randn(2,4,5,5)
        for m in (global_m,spatial):
            actual,w,e=m.components(q,x)
            self.assertTrue(torch.allclose(w.sum(1),torch.ones_like(w[:,0])))
            expected=sum(w[:,k:k+1]*(e[:,k,:,None,None]*x).sum(1,keepdim=True) for k in range(4))
            torch.testing.assert_close(actual,expected)
            if m.mode=='global_mix':self.assertTrue(torch.equal(w,w[:,:,:1,:1].expand_as(w)))
            else:self.assertGreater(float(w.var((-2,-1)).sum()),0)
        self.assertFalse(torch.allclose(global_m(q,x),spatial(q,x)))

    def test_explicit_checkpoint_arguments_preserve_gradients(self):
        torch.manual_seed(44);a=SpatialQuery(4,4,4)
        torch.nn.init.normal_(a.experts[-1].weight);b=copy.deepcopy(a)
        q,x=torch.randn(2,4,requires_grad=True),torch.randn(2,4,5,5,requires_grad=True)
        qb,xb=q.detach().clone().requires_grad_(),x.detach().clone().requires_grad_()
        first=a(q,x);second=checkpoint(b,qb,xb,use_reentrant=False,preserve_rng_state=True)
        self.assertTrue(torch.equal(first,second));first.sum().backward();second.sum().backward()
        self.assertTrue(torch.equal(q.grad,qb.grad));self.assertTrue(torch.equal(x.grad,xb.grad))
        for p,z in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(p.grad,z.grad))

    def test_production_pause_resume_with_new_parameters(self):
        namespace=old.make_namespace();logger=namespace['logger_config']
        tree=ast.parse((ROOT/'train_spatial.py').read_text(encoding='utf-8'))
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))],type_ignores=[]),'train_spatial production','exec'),namespace)
        namespace['logger_config']=logger;original=old.make_cfg
        def dataset(cfg,split,records):
            return PromptDataset(old.ToyDataset(split.lower(),'toy',[],old.ToyTransform(augment=split=='Train')),split=='Val')
        for mode in MODES:
            def config():
                cfg=original();cfg.MODEL.VARIANT='paired_image';cfg.TRAIN.DICE_MODE='image'
                cfg.TRAIN.NEW_LEARNING_RATE=.003;cfg.TRAIN.MIN_LEARNING_RATE=.0001
                cfg.RESEARCH=SimpleNamespace(BASE_WEIGHTS='toy',RUN_ID=mode+'_seed666',PROMPT_MODE='generic',QUERY_MODE=mode,GENERIC_PROBABILITY=1.)
                return cfg
            with self.subTest(mode=mode),patch.object(old,'make_cfg',config), \
                 patch('research_spatial.model.build_model',lambda cfg:ToySpatial(cfg.RESEARCH.QUERY_MODE)), \
                 patch('research_spatial.common.make_signature',lambda cfg:{'digest':'spatial-toy'}), \
                 patch('research_spatial.data.make_dataset',dataset),tempfile.TemporaryDirectory() as directory:
                old.test_repeated_resume(namespace,Path(directory))

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main(verbosity=2)
