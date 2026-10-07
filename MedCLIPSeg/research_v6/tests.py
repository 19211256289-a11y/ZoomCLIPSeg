"""Prompt/geometry, gradient, paired-objective and production pause regressions."""
from local_runtime import configure_local_models
configure_local_models()
import ast,copy,importlib.util,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from train_v6 import set_random_seed,forward_batch,calc_loss
from .objective import reflect_text,reflected_batch,cross_view_loss,paired_objective
from .common import ROOT,WORKSPACE
from utils.main_utils import read_text
spec=importlib.util.spec_from_file_location('reference_pause',WORKSPACE/'reports/test_training_pause.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

class V6Tests(unittest.TestCase):
    def test_prompt_and_tensor_reflections(self):
        for split in ('Train','Val'):
            for row in read_text(ROOT/f'data/Kvasir/Prompts_Folder/{split}_text.xlsx'):
                text=row['Description']
                for d in (-1,-2):
                    reflected=reflect_text(text,d)
                    self.assertEqual(reflect_text(reflected,d),text)
                    self.assertEqual(reflected.split('located in ')[0],text.split('located in ')[0])
        x=torch.arange(12).reshape(1,1,3,4)
        batch=dict(image=x,ground_truth_mask=x[:,0],text_prompt=['two brown polyps, located in top left, bottom right of the image'])
        for d in (-1,-2):
            b=reflected_batch(batch,d);old.assert_equal(reflected_batch(b,d),batch)
            self.assertTrue(torch.equal(b['image'][:,0],b['ground_truth_mask']))
        self.assertIn('top right, bottom left',reflected_batch(batch,-1)['text_prompt'][0])

    def test_saturated_wrong_student_receives_gradient(self):
        a=torch.tensor([[[100.,-100.]]],requires_grad=True)
        b=torch.tensor([[[-100.,100.]]],requires_grad=True)
        loss,stats=cross_view_loss(a,b,torch.tensor([[[1.,0.]]]))
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertGreater(float(loss),90)
        self.assertEqual(float(a.grad.abs().sum()),0)
        self.assertLess(float(b.grad[0,0,0]),0);self.assertGreater(float(b.grad[0,0,1]),0)
        self.assertEqual(stats['teacher_valid_fraction'],1.)

    def test_both_wrong_empty_full_and_identical(self):
        for y in (torch.zeros(1,2,2),torch.ones(1,2,2)):
            a=((1-2*y)*10).requires_grad_();b=((1-2*y)*20).requires_grad_()
            loss,stats=cross_view_loss(a,b,y);loss.backward()
            self.assertEqual(float(loss),0);self.assertEqual(stats['both_wrong_fraction'],1.)
            self.assertEqual(float(a.grad.abs().sum()+b.grad.abs().sum()),0)
        a=torch.tensor([[[2.,-3.]]],requires_grad=True)
        loss,_=cross_view_loss(a,a,torch.tensor([[[1.,0.]]]));self.assertLess(abs(float(loss)),1e-6)

    def test_zero_weight_exact_paired_control(self):
        c=old.make_cfg();c.TRAIN.DICE_MODE='image';c.TRAIN.CONSISTENCY_WEIGHT=0.
        batch=old.ToyDataset('train','toy',[],old.ToyTransform())[0]
        batch={k:(v[None] if isinstance(v,torch.Tensor) else ['one polyp, located in top left of the image']) for k,v in batch.items()}
        set_random_seed(46);initial=old.ToyModel();states=[]
        for variant in ('paired_aug','paired_consistency'):
            c.MODEL.VARIANT=variant;m=copy.deepcopy(initial);o=torch.optim.Adam(m.parameters(),lr=.003)
            set_random_seed(666);loss,_=paired_objective(m,batch,c,torch.nn.BCEWithLogitsLoss(),old.ToyDiceLoss(),forward_batch,calc_loss)
            loss.backward();o.step();states.append(dict(model=m.state_dict(),optimizer=o.state_dict(),loss=loss.detach()))
        old.assert_equal(*states)

    def test_actual_paired_pause_resume(self):
        namespace=old.make_namespace();logger=namespace['logger_config']
        parsed=ast.parse((ROOT/'train_v6.py').read_text(encoding='utf-8'))
        exec(compile(ast.Module(body=[n for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))],type_ignores=[]),'train_v6 production functions','exec'),namespace)
        namespace['logger_config']=logger;original_cfg=old.make_cfg
        def cfg():
            c=original_cfg();c.MODEL.VARIANT='paired_consistency';c.TRAIN.DICE_MODE='image';c.TRAIN.CONSISTENCY_WEIGHT=.1
            c.TRAIN.NEW_LEARNING_RATE=.003;c.TRAIN.MIN_LEARNING_RATE=.0001;c.RESEARCH=SimpleNamespace(INIT_FROM='toy');return c
        class Data(old.ToyDataset):
            def __getitem__(self,i):
                b=super().__getitem__(i);b['text_prompt']='one polyp, located in top left of the image';return b
        def data(c,split,records):return Data(split.lower(),'toy',[],old.ToyTransform(augment=split=='Train'))
        with patch.object(old,'make_cfg',cfg),patch('research_v6.model.build_model',lambda cfg:old.ToyModel()), \
             patch('research_v6.model.initialize_from_checkpoint',lambda *a:[]), \
             patch('research_v6.common.make_signature',lambda cfg:{'digest':'v6-toy'}), \
             patch('research_v6.data.make_dataset',data),tempfile.TemporaryDirectory() as temp:
            old.test_repeated_resume(namespace,Path(temp))

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main(verbosity=2)
