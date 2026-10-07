"""Intervention semantics and restart determinism, using the production iterator."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
from .prompts import CONDITIONS, GENERIC, prompt_for, intervention_batch, align_predictions, direction_bearing


class ToyDataset(Dataset):
    def __len__(self):
        return 5
    def __getitem__(self,index):
        return dict(image=torch.arange(48).reshape(3,4,4).float()/48+index,
                    ground_truth_mask=torch.zeros(4,4,dtype=torch.long),
                    text_prompt='one small polyp, located in top left of the image',
                    image_name=f'{index}.png',mask_name=f'{index}.png')


class ToyModel(torch.nn.Module):
    def forward(self,image,text,num_samples):
        return image[:,0].unsqueeze(0)+torch.randn(num_samples,image.shape[0],4,4)*.1


class ValidationTests(unittest.TestCase):
    def test_generic_has_no_sample_dependency(self):
        for text in (None,'any private attributes','one polyp, located in center of the image'):
            self.assertEqual(prompt_for(text,'generic'),GENERIC)

    def test_location_removal_and_opposite_semantics(self):
        text='one small rounded polyp, located in top left of the image'
        self.assertEqual(prompt_for(text,'no_location'),'one small rounded polyp')
        self.assertEqual(prompt_for(text,'opposite_location'),'one small rounded polyp, located in bottom right of the image')
        self.assertEqual(prompt_for(prompt_for(text,'opposite_location'),'opposite_location'),text)
        centered='one small polyp, located in center of the image'
        self.assertEqual(prompt_for(centered,'opposite_location'),centered)
        self.assertFalse(direction_bearing(centered))
        self.assertTrue(direction_bearing(text))
        with self.assertRaises(ValueError):
            prompt_for('unsupported','no_location')
        with self.assertRaises(ValueError):
            prompt_for(text,'unlocked_condition')

    def test_spatial_inverse_and_no_gt_modification(self):
        sample=ToyDataset()[0]
        batch=dict(sample,image=sample['image'].unsqueeze(0),text_prompt=[sample['text_prompt']])
        for condition,dim in [('hflip_sync',-1),('vflip_sync',-2)]:
            view,inverse=intervention_batch(batch,condition)
            self.assertEqual(dim,inverse)
            self.assertTrue(torch.equal(view['image'],batch['image'].flip(dim)))
            self.assertIs(view['ground_truth_mask'],batch['ground_truth_mask'])
            oracle=batch['image'][:,0].unsqueeze(0)
            self.assertTrue(torch.equal(align_predictions(oracle.flip(dim),inverse),oracle))
        unchanged,dim=intervention_batch(batch,'original')
        self.assertIs(unchanged['image'],batch['image'])
        self.assertIsNone(dim)
        self.assertEqual(unchanged['text_prompt'],batch['text_prompt'])

    def test_actual_iterator_pause_and_restart_matches_uninterrupted(self):
        from .runner import infer_batches
        from train_formal import TrainingControl,TrainingPaused
        cfg=SimpleNamespace(seed=667,MODEL=SimpleNamespace(DEVICE='cpu'))
        with tempfile.TemporaryDirectory() as directory:
            control=TrainingControl(directory,phase='unit_test')
            uninterrupted=list(infer_batches(ToyModel(),ToyDataset(),cfg,'hflip_sync',control,samples=3,batch_size=2))
            interrupted=infer_batches(ToyModel(),ToyDataset(),cfg,'hflip_sync',control,samples=3,batch_size=2)
            first=next(interrupted)
            for x,y in zip(first[1],uninterrupted[0][1]):
                np.testing.assert_array_equal(x,y)
            pause=Path(directory)/'PAUSE.request'
            pause.write_text('requested')
            with self.assertRaises(TrainingPaused):
                next(interrupted)
            pause.unlink()
            resumed=list(infer_batches(ToyModel(),ToyDataset(),cfg,'hflip_sync',control,samples=3,batch_size=2))
            self.assertEqual(len(resumed),3)
            for a,b in zip(uninterrupted,resumed):
                self.assertEqual(a[0]['mask_name'],b[0]['mask_name'])
                for x,y in zip(a[1],b[1]):
                    np.testing.assert_array_equal(x,y)

    def test_cached_artifact_tampering_is_rejected(self):
        from .runner import completed_summary
        from research_formal.common import write_json,sha256
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            write_json(p/'summary.json',{'n':100})
            write_json(p/'manifest.json',dict(complete=True,signature={'condition':'original'},outputs={'summary.json':sha256(p/'summary.json')}))
            self.assertEqual(completed_summary(p,{'condition':'original'})['n'],100)
            with self.assertRaises(ValueError):
                completed_summary(p,{'condition':'generic'})
            write_json(p/'summary.json',{'n':99})
            with self.assertRaises(ValueError):
                completed_summary(p,{'condition':'original'})


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
