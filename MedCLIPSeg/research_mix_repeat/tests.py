"""Reuse unchanged model tests and exercise the new seed protocol and trainer."""
from local_runtime import configure_local_models
configure_local_models()
import ast,copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from research_spatial.tests import SpatialTests,ToySpatial
from research_formal.tests import old,toy_batch
from train_mix_repeat import set_random_seed,capture_rng_state
from .common import ROOT,MODES,load_suite,make_config,split_run
from .data import PromptDataset
from .objective import prompt_views
from .gate import replication_gate

class RepeatTests(unittest.TestCase):
    def test_distinct_seeds_matched_views_and_paths(self):
        suite=load_suite();self.assertEqual(len(suite['runs']),4)
        for seed in suite['seeds']:
            views=[];rng=[]
            for mode in MODES:
                cfg=make_config(suite,f'{mode}_seed{seed}')
                self.assertEqual(cfg.seed,seed);self.assertEqual(cfg.RESEARCH.INITIALIZATION_SEED,seed)
                self.assertEqual(cfg.RESEARCH.QUERY_MODE,mode);self.assertFalse(cfg.resume);self.assertIsNone(cfg.init_from)
                set_random_seed(seed);views.append(prompt_views(toy_batch(),cfg)[:2]);rng.append(capture_rng_state())
            old.assert_equal(views[0],views[1]);old.assert_equal(rng[0],rng[1])
        with self.assertRaises(ValueError):split_run('global_mix_seed666')

    def test_replication_rejects_one_seed_or_recall_tradeoff(self):
        suite=load_suite();summaries={};cases={}
        for seed in suite['seeds']:
            b,t=f'baseline_seed{seed}',f'global_mix_seed{seed}'
            summaries[b]=dict(DSC_percent=87.,small_DSC_percent=84.,micro_foreground_precision_percent=89.,micro_foreground_recall_percent=84.,zero_overlap_positive=0)
            summaries[t]=dict(summaries[b]);summaries[t]['micro_foreground_precision_percent']+=.6
            cases[b]=[dict(name=str(i),gt_pixels=100,fp_pixels=20,DSC=.87) for i in range(4)]
            cases[t]=[dict(name=str(i),gt_pixels=100,fp_pixels=15,DSC=.87) for i in range(4)]
        self.assertTrue(replication_gate(summaries,cases,suite)['passed'])
        summaries['global_mix_seed668']['micro_foreground_precision_percent']=88.9
        self.assertFalse(replication_gate(summaries,cases,suite)['passed'])
        summaries['global_mix_seed668']['micro_foreground_precision_percent']=89.6
        summaries['global_mix_seed667']['micro_foreground_recall_percent']=82.0
        self.assertFalse(replication_gate(summaries,cases,suite)['passed'])

    def test_actual_new_trainer_pause_resume_both_seeds(self):
        namespace=old.make_namespace();logger=namespace['logger_config']
        tree=ast.parse((ROOT/'train_mix_repeat.py').read_text(encoding='utf-8'))
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))],type_ignores=[]),'train_mix_repeat production','exec'),namespace)
        namespace['logger_config']=logger;original=old.make_cfg
        def dataset(cfg,split,records):
            return PromptDataset(old.ToyDataset(split.lower(),'toy',[],old.ToyTransform(augment=split=='Train')),split=='Val')
        for run in load_suite()['runs']:
            mode,seed=split_run(run)
            def config():
                cfg=original();cfg.seed=seed;cfg.MODEL.VARIANT='paired_image';cfg.TRAIN.DICE_MODE='image'
                cfg.TRAIN.NEW_LEARNING_RATE=.003;cfg.TRAIN.MIN_LEARNING_RATE=.0001
                cfg.RESEARCH=SimpleNamespace(BASE_WEIGHTS='toy',RUN_ID=run,PROMPT_MODE='generic',QUERY_MODE=mode,GENERIC_PROBABILITY=1.)
                return cfg
            with self.subTest(run=run),patch.object(old,'make_cfg',config), \
                 patch('research_mix_repeat.model.build_model',lambda cfg:ToySpatial(cfg.RESEARCH.QUERY_MODE)), \
                 patch('research_mix_repeat.common.make_signature',lambda cfg:{'digest':'repeat-toy'}), \
                 patch('research_mix_repeat.data.make_dataset',dataset),tempfile.TemporaryDirectory() as directory:
                old.test_repeated_resume(namespace,Path(directory))

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main(verbosity=2)
