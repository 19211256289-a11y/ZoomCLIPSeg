"""Dual-input geometry, explicit recomputation and production pause regression."""
from local_runtime import configure_local_models
configure_local_models()
import ast,copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.utils.checkpoint import checkpoint
from research_formal.tests import old,toy_batch
from research_spatial.tests import ToySpatial
from research_spatial.model import SpatialCLIP,SpatialQuery
from research.modules import DetailEncoder,DetailDecoder
from research.data import DualScaleDataset,DualScaleTransform
from research_formal.data import make_dataset as original_dataset
from utils.main_utils import read_text
from train_detail_formal import set_random_seed,capture_rng_state
from .common import ROOT,MODES,GENERIC,load_suite,make_config
from .data import make_dataset,PromptDataset
from .objective import prompt_views
from .model import DetailMixCLIP
from .gate import combination_gate


class ToyCombo(ToySpatial):
    def __init__(self,mode):
        super().__init__('global_mix' if mode in ('global_mix','detail_global_mix') else 'baseline')
        self.has_detail=mode in ('detail','detail_global_mix')
        if self.has_detail:
            self.detail_encoder=torch.nn.Conv2d(1,2,1)
            self.detail_decoder=torch.nn.Conv2d(2,1,1)
            torch.nn.init.zeros_(self.detail_decoder.weight);torch.nn.init.zeros_(self.detail_decoder.bias)
    def forward(self,image,text,num_samples=1,image_detail=None):
        raw=super().forward(image,text,num_samples)
        if not self.has_detail:return raw
        extra=self.detail_decoder(self.detail_encoder(image_detail)).squeeze(1)
        return (raw[0]+extra,raw[1]) if self.training else raw+extra.unsqueeze(0)


class CombinationTests(unittest.TestCase):
    def test_host_assignment_and_portable_paths(self):
        from .common import assigned_runs,current_host,portable,WORKSPACE
        suite=load_suite()
        import os
        seen=[]
        for host,seeds in (('local',{666}),('server4090',{667,668})):
            with patch.dict(os.environ,{'MEDCLIPSEG_HOST':host}):
                runs=assigned_runs(suite);seen.extend(runs)
                self.assertEqual({int(r.rsplit('_seed',1)[1]) for r in runs},seeds)
                self.assertEqual(current_host(),host)
        self.assertEqual(sorted(seen),sorted(suite['runs']))
        self.assertEqual(portable(str(WORKSPACE/'MedCLIPSeg/data')),'${WORKSPACE}/MedCLIPSeg/data')

    def test_real_dual_data_preserves_224_and_rng(self):
        cfg=make_config(load_suite(),'detail_seed666')
        records=read_text(ROOT/'data/Kvasir/Prompts_Folder/Train_text.xlsx')[:3]
        source=original_dataset(cfg,'Train',records);dual=make_dataset(cfg,'Train',records)
        fake=DualScaleDataset(cfg.DATASET.TRAIN_PATH,'Kvasir',records,DualScaleTransform(True),image_size=224,detail_size=448,upsampled=True)
        for i in range(3):
            for seed in (1,2):
                set_random_seed(seed);a=source[i];rng=capture_rng_state()
                set_random_seed(seed);b=dual[i];old.assert_equal(rng,capture_rng_state())
                for key in ('image','ground_truth_mask'):self.assertTrue(torch.equal(a[key],b[key]))
                self.assertEqual(b['image_detail'].shape,(3,448,448))
                set_random_seed(seed);c=fake[i]
                self.assertFalse(torch.equal(b['image_detail'],c['image_detail']))

    def test_both_reflections_align_all_three_tensors(self):
        batch=toy_batch();batch['image_detail']=batch['image'].repeat_interleave(2,-1).repeat_interleave(2,-2)
        for dim in (-1,-2):
            outputs=[];states=[]
            for mode in MODES:
                cfg=make_config(load_suite(),mode+'_seed666');set_random_seed(19)
                with patch('research_detail_formal.objective.random.choice',return_value=dim):a,b,_=prompt_views(batch,cfg)
                for key in ('image','image_detail','ground_truth_mask'):
                    self.assertTrue(torch.equal(b[key],a[key].flip(dim)))
                self.assertEqual(a['text_prompt'],[GENERIC]);self.assertEqual(b['text_prompt'],[GENERIC])
                outputs.append((a,b));states.append(capture_rng_state())
            for i in range(1,4):old.assert_equal(outputs[0],outputs[i]);old.assert_equal(states[0],states[i])

    def test_zero_detail_preserves_nonzero_global_module(self):
        m=DetailMixCLIP.__new__(DetailMixCLIP);torch.nn.Module.__init__(m)
        m.patch_size=16;m.im_size=224;m.query_mode='global_mix';m.detail_checkpointing=True
        m.mask_head=torch.nn.Linear(4,4);m.upscale=torch.nn.Upsample(scale_factor=4,mode='bilinear',align_corners=False)
        m.spatial_query=SpatialQuery(4,4,4,4,'global_mix');torch.nn.init.normal_(m.spatial_query.experts[-1].weight)
        m.detail_decoder=DetailDecoder(4);m.eval()
        image,text=torch.randn(1,197,4),torch.randn(1,4)
        detail=(torch.randn(1,16,224,224),torch.randn(1,32,112,112),torch.randn(1,64,56,56))
        actual=m.segment_with_detail(image,text,1,224,224,detail)
        expected=SpatialCLIP.compute_seg_logits(m,image,text,1,224,224)[0]
        self.assertTrue(torch.equal(actual,expected))

    def test_explicit_detail_checkpoint_gradients(self):
        torch.manual_seed(42);encoder=DetailEncoder();decoder=DetailDecoder(4)
        torch.nn.init.normal_(decoder.out.weight,std=.01)
        image=torch.randn(1,3,448,448);features=torch.randn(1,4,56,56)
        results=[]
        for enabled in (False,True):
            encoder.zero_grad();decoder.zero_grad();x=features.clone().requires_grad_()
            if enabled:
                detail=checkpoint(encoder,image,use_reentrant=False,preserve_rng_state=True)
                z=checkpoint(decoder,x,detail,use_reentrant=False,preserve_rng_state=True)
            else:z=decoder(x,encoder(image))
            z.square().mean().backward()
            results.append([z.detach(),x.grad.clone()]+[p.grad.clone() for p in list(encoder.parameters())+list(decoder.parameters())])
        for a,b in zip(*results):torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_gate_distinguishes_small_gain_and_precision_tradeoff(self):
        s=dict(DSC_percent=87.,NSD_percent=89.,small_DSC_percent=84.,micro_foreground_precision_percent=89.,
            micro_foreground_recall_percent=85.,positive_DSC_below_50_percent=4,zero_overlap_positive=1,source_train_area_threshold=.06)
        summaries={m:dict(s) for m in MODES};summaries['detail_global_mix']['small_DSC_percent']+=.4
        cases={m:[dict(name=str(i),DSC=.8,gt_fraction=.02) for i in range(4)] for m in MODES}
        for r in cases['detail_global_mix']:r['DSC']=.804
        self.assertTrue(combination_gate(summaries,cases,load_suite())['passed'])
        summaries['detail_global_mix']['micro_foreground_precision_percent']=88.
        self.assertFalse(combination_gate(summaries,cases,load_suite())['passed'])
        cases['detail'][0]['name']='different'
        with self.assertRaises(ValueError):combination_gate(summaries,cases,load_suite())

    def test_all_modes_source_only_and_equal_budget(self):
        suite=load_suite();self.assertEqual(len(suite['runs']),12)
        for mode in MODES:
            cfg=make_config(suite,mode+'_seed666')
            self.assertEqual(cfg.TRAIN.NUM_EPOCHS,100);self.assertEqual(cfg.TRAIN.BATCH_SIZE,24)
            self.assertEqual(cfg.TRAIN.NEW_LEARNING_RATE,cfg.TRAIN.LEARNING_RATE)
            self.assertTrue(cfg.RESEARCH.DETAIL_CHECKPOINTING);self.assertEqual(cfg.seed,666)
            with self.assertRaises(ValueError):make_dataset(cfg,'Test',[])

    def test_actual_trainer_repeated_pause_resume_all_four_modes(self):
        namespace=old.make_namespace();logger=namespace['logger_config']
        tree=ast.parse((ROOT/'train_detail_formal.py').read_text(encoding='utf-8'))
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))],type_ignores=[]),'train_detail_formal production','exec'),namespace)
        namespace['logger_config']=logger;original=old.make_cfg
        class ToyDataset(old.ToyDataset):
            def __getitem__(self,index):
                sample=super().__getitem__(index);sample['image_detail']=sample['image'].clone();return sample
        def dataset(cfg,split,records):
            return PromptDataset(ToyDataset(split.lower(),'toy',[],old.ToyTransform(augment=split=='Train')),split=='Val')
        for mode in MODES:
            def config():
                cfg=original();cfg.MODEL.VARIANT='paired_image';cfg.TRAIN.DICE_MODE='image'
                cfg.TRAIN.NEW_LEARNING_RATE=.003;cfg.TRAIN.MIN_LEARNING_RATE=.0001
                cfg.RESEARCH=SimpleNamespace(BASE_WEIGHTS='toy',RUN_ID=mode+'_seed666',PROMPT_MODE='generic',QUERY_MODE=mode,GENERIC_PROBABILITY=1.)
                return cfg
            with self.subTest(mode=mode),patch.object(old,'make_cfg',config), \
                patch('research_detail_formal.model.build_model',lambda cfg:ToyCombo(cfg.RESEARCH.QUERY_MODE)), \
                patch('research_detail_formal.common.make_signature',lambda cfg:{'digest':'detail-mix-toy'}), \
                patch('research_detail_formal.data.make_dataset',dataset),tempfile.TemporaryDirectory() as directory:
                old.test_repeated_resume(namespace,Path(directory))

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main(verbosity=2)
