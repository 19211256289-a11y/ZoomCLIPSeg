"""Prompt semantics, matched RNG and production mid-batch pause/resume."""
from local_runtime import configure_local_models
configure_local_models()
import ast,copy,random,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from research_formal.tests import old,toy_batch
from train_softcon import set_random_seed,capture_rng_state,forward_batch,calc_loss
from .common import ROOT,MODES,GENERIC,load_suite,make_config
from .data import PromptDataset
from .objective import prompt_views,paired_objective

class PromptTests(unittest.TestCase):
    def test_weighted_reference_objective_and_zero_weight_gradients(self):
        from research_prompt.objective import paired_objective as old_objective
        for mode in MODES:
            for weight in (0.,.1):
                cfg=old.make_cfg();cfg.TRAIN.DICE_MODE='image';cfg.TRAIN.CLIP_WEIGHT=weight
                cfg.RESEARCH=SimpleNamespace(PROMPT_MODE=mode,GENERIC_PROBABILITY=1. if mode=='generic' else .5)
                initial=old.ToyModel();values=[]
                for objective in (old_objective,paired_objective):
                    model=copy.deepcopy(initial);set_random_seed(666)
                    loss,_=objective(model,toy_batch(),cfg,torch.nn.BCEWithLogitsLoss(),old.ToyDiceLoss(),forward_batch,calc_loss)
                    loss.backward();values.append(dict(loss=loss.detach(),grads=[p.grad for p in model.parameters()],rng=capture_rng_state()))
                old.assert_equal(values[0],values[1])
        suite=load_suite()
        self.assertTrue(all(make_config(suite,r).TRAIN.CLIP_WEIGHT==0 for r in suite['runs']))

    def test_prompt_modes_share_geometry_and_rng(self):
        states,views=[],[]
        batch=toy_batch()
        batch={k:(v.repeat(24,*([1]*(v.ndim-1))) if torch.is_tensor(v) else v*24) for k,v in batch.items()}
        for mode in MODES:
            cfg=make_config(load_suite(),mode+'_seed666')
            set_random_seed(666)
            first,second,diag=prompt_views(batch,cfg)
            states.append(capture_rng_state());views.append((first,second))
            self.assertEqual(first['text_prompt'],second['text_prompt'])
            if mode=='generic':self.assertEqual(first['text_prompt'],[GENERIC]*24)
            if mode=='attribute':self.assertEqual(first['text_prompt'],batch['text_prompt'])
            if mode=='mixed':self.assertTrue(0<diag['generic_count']<24)
        for index in (1,):
            old.assert_equal(states[0],states[index])
            for key in ('image','ground_truth_mask'):
                self.assertTrue(torch.equal(views[0][1][key],views[index][1][key]))
        self.assertNotEqual(batch['text_prompt'],[GENERIC]*24)

    def test_validation_class_sentence_does_not_use_attributes(self):
        source=[dict(image=torch.ones(3,4,4),ground_truth_mask=torch.zeros(4,4),text_prompt='secret annotation')]
        data=PromptDataset(source,True)
        self.assertEqual(data[0]['text_prompt'],GENERIC)
        self.assertEqual(source[0]['text_prompt'],'secret annotation')
        self.assertTrue(torch.equal(data[0]['image'],source[0]['image']))

    def test_manual_loss_and_gradients(self):
        for mode in MODES:
            cfg=old.make_cfg();cfg.TRAIN.DICE_MODE='image';cfg.TRAIN.CLIP_WEIGHT=0.
            cfg.RESEARCH=SimpleNamespace(PROMPT_MODE=mode,GENERIC_PROBABILITY={'attribute':0.,'generic':1.,'mixed':.5}[mode])
            actual=old.ToyModel();manual=copy.deepcopy(actual);batch=toy_batch()
            set_random_seed(666)
            loss,_=paired_objective(actual,batch,cfg,torch.nn.BCEWithLogitsLoss(),old.ToyDiceLoss(),forward_batch,calc_loss)
            loss.backward();state=capture_rng_state()
            set_random_seed(666)
            first,second,_=prompt_views(batch,cfg);parts=[]
            for view in (first,second):
                z,con=forward_batch(manual,view,'cpu')
                parts.append(calc_loss(z,view['ground_truth_mask'],torch.nn.BCEWithLogitsLoss(),old.ToyDiceLoss(),cfg)+cfg.TRAIN.CLIP_WEIGHT*con)
            expected=sum(parts)/2;expected.backward()
            self.assertTrue(torch.equal(loss,expected));old.assert_equal(state,capture_rng_state())
            for p,q in zip(actual.parameters(),manual.parameters()):self.assertTrue(torch.equal(p.grad,q.grad))

    def test_production_pause_resume_all_prompt_modes(self):
        namespace=old.make_namespace();logger=namespace['logger_config']
        tree=ast.parse((ROOT/'train_softcon.py').read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'train_softcon production','exec'),namespace)
        namespace['logger_config']=logger
        original_cfg=old.make_cfg
        class Data(old.ToyDataset):
            def __getitem__(self,index):
                result=super().__getitem__(index);result['text_prompt']='one polyp, located in top left of the image'
                return result
        def dataset(cfg,split,records):return PromptDataset(Data(split.lower(),'toy',[],old.ToyTransform(augment=split=='Train')),split=='Val')
        for mode in MODES:
            def config():
                cfg=original_cfg();cfg.MODEL.VARIANT='paired_image';cfg.TRAIN.DICE_MODE='image';cfg.TRAIN.CLIP_WEIGHT=0.
                cfg.TRAIN.NEW_LEARNING_RATE=.003;cfg.TRAIN.MIN_LEARNING_RATE=.0001
                cfg.RESEARCH=SimpleNamespace(BASE_WEIGHTS='toy',RUN_ID=mode+'_seed666',PROMPT_MODE=mode,
                                            GENERIC_PROBABILITY={'attribute':0.,'generic':1.,'mixed':.5}[mode])
                return cfg
            with self.subTest(mode=mode),patch.object(old,'make_cfg',config), \
                 patch('research_formal.model.build_model',lambda cfg:old.ToyModel()), \
                 patch('research_softcon.common.make_signature',lambda cfg:{'digest':'prompt-toy'}), \
                 patch('research_softcon.data.make_dataset',dataset),tempfile.TemporaryDirectory() as directory:
                old.test_repeated_resume(namespace,Path(directory))

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
