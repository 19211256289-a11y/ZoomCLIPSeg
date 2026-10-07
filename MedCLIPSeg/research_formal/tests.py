"""Matched-view semantics, RNG order, seed isolation and production pause tests."""
from local_runtime import configure_local_models
configure_local_models()
import ast, copy, importlib.util, random, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from train_formal import set_random_seed, forward_batch, calc_loss, capture_rng_state
from .objective import reflect_text, reflected_batch, second_view, paired_objective
from .common import ROOT, WORKSPACE, VARIANTS, load_suite, make_config, checkpoint_path
from .report import paired_comparison
spec = importlib.util.spec_from_file_location('reference_pause', WORKSPACE/'reports/test_training_pause.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def toy_batch():
    return dict(image=torch.arange(12).reshape(1, 1, 3, 4).float()/12,
                ground_truth_mask=torch.tensor([[[1, 0, 0, 1], [0, 1, 0, 0], [1, 1, 0, 1]]]),
                text_prompt=['two brown polyps, located in top left, bottom right of the image'])


class FormalTests(unittest.TestCase):
    def test_prompt_and_view_semantics(self):
        from utils.main_utils import read_text
        for split in ('Train', 'Val'):
            for row in read_text(ROOT/f'data/Kvasir/Prompts_Folder/{split}_text.xlsx'):
                for dim in (-1, -2):
                    text = row['Description']
                    self.assertEqual(reflect_text(reflect_text(text, dim), dim), text)
                    self.assertEqual(reflect_text(text, dim).split('located in ')[0], text.split('located in ')[0])
        batch = toy_batch()
        for dim in (-1, -2):
            old.assert_equal(second_view(batch, 'paired_repeat', dim), batch)
            a, b = [second_view(batch, v, dim) for v in ('paired_image', 'paired_aug')]
            self.assertTrue(torch.equal(a['image'], batch['image'].flip(dim)))
            self.assertTrue(torch.equal(a['ground_truth_mask'], batch['ground_truth_mask'].flip(dim)))
            self.assertTrue(torch.equal(a['image'], b['image']))
            self.assertEqual(a['text_prompt'], batch['text_prompt'])
            self.assertNotEqual(b['text_prompt'], batch['text_prompt'])
            old.assert_equal(reflected_batch(reflected_batch(batch, dim), dim), batch)

    def test_loss_gradients_and_rng_match_manual_supervision(self):
        batch = toy_batch()
        set_random_seed(12)
        initial = old.ToyModel()
        rng_states = []
        for variant in VARIANTS:
            cfg = old.make_cfg()
            cfg.TRAIN.DICE_MODE = 'image'
            cfg.MODEL.VARIANT = variant
            actual, manual = copy.deepcopy(initial), copy.deepcopy(initial)
            set_random_seed(667)
            loss, diagnostics = paired_objective(actual, batch, cfg, torch.nn.BCEWithLogitsLoss(), old.ToyDiceLoss(), forward_batch, calc_loss)
            loss.backward()
            rng_states.append(capture_rng_state())
            set_random_seed(667)
            dim = random.choice((-1, -2))
            parts = []
            for view in (batch, second_view(batch, variant, dim)):
                logits, con = forward_batch(manual, view, 'cpu')
                parts.append(calc_loss(logits, view['ground_truth_mask'], torch.nn.BCEWithLogitsLoss(), old.ToyDiceLoss(), cfg) + cfg.TRAIN.CLIP_WEIGHT*con)
            expected = sum(parts)/2
            expected.backward()
            self.assertTrue(torch.equal(loss, expected))
            for p, q in zip(actual.parameters(), manual.parameters()):
                self.assertTrue(torch.equal(p.grad, q.grad))
            self.assertEqual(diagnostics['forward_backward_passes'], 2)
            self.assertEqual(diagnostics['consistency_weight'], 0)
        old.assert_equal(rng_states[0], rng_states[1])
        old.assert_equal(rng_states[1], rng_states[2])

    def test_synced_arm_preserves_v6_objective(self):
        from research_v6.objective import paired_objective as previous
        cfg = old.make_cfg()
        cfg.TRAIN.DICE_MODE = 'image'
        cfg.MODEL.VARIANT = 'paired_aug'
        batch = toy_batch()
        set_random_seed(40)
        initial = old.ToyModel()
        states = []
        for objective in (previous, paired_objective):
            model = copy.deepcopy(initial)
            set_random_seed(666)
            loss, _ = objective(model, batch, cfg, torch.nn.BCEWithLogitsLoss(), old.ToyDiceLoss(), forward_batch, calc_loss)
            loss.backward()
            states.append(dict(loss=loss.detach(), gradients=[p.grad for p in model.parameters()], rng=capture_rng_state()))
        old.assert_equal(*states)

    def test_seed_outputs_and_incomplete_report(self):
        suite = load_suite('formal_kvasir_v1')
        self.assertEqual(len(suite['runs']), 9)
        self.assertEqual(len({checkpoint_path(suite, run) for run in suite['runs']}), 9)
        configs = [make_config(suite, run, True) for run in suite['runs']]
        self.assertEqual(len({c.RESEARCH.BASE_WEIGHTS for c in configs}), 1)
        self.assertTrue(all('outputs_' not in c.RESEARCH.BASE_WEIGHTS for c in configs))
        self.assertTrue(all(c.RESEARCH.INITIALIZATION_SEED == c.seed for c in configs))
        self.assertEqual({c.seed for c in configs}, {666, 667, 668})
        self.assertTrue(all(c.TRAIN.NUM_EPOCHS == 100 and c.TRAIN.BATCH_SIZE == 24 for c in configs))
        self.assertTrue(all(c.TRAIN.LEARNING_RATE == .0003 and c.TRAIN.MIN_LEARNING_RATE == .0001 for c in configs))
        self.assertTrue(all(c.TRAIN.CONSISTENCY_WEIGHT == 0 and c.MODEL.GRAD_CHECKPOINTING for c in configs))
        comparison = paired_comparison({}, 'paired_aug', 'paired_repeat', suite['seeds'])
        self.assertFalse(comparison['complete'])
        from .runner import final_evaluation_jobs, require_all_final_checkpoints
        self.assertEqual(len(final_evaluation_jobs(suite)), 45)
        incomplete = dict(epoch=98, progress=dict(phase='epoch_complete'), research_signature=dict(digest='test'))
        plan = dict(signatures={run: dict(digest='test') for run in suite['runs']})
        with patch('torch.load', return_value=incomplete), self.assertRaises(ValueError):
            require_all_final_checkpoints(suite, True, plan)
        incomplete['epoch'] = 99
        with patch('torch.load', return_value=incomplete) as mocked:
            require_all_final_checkpoints(suite, True, plan)
            self.assertEqual(mocked.call_count, 9)

    def test_base_loading_is_strict_and_warm_start_rejected(self):
        from .model import strict_base_state
        from .common import make_signature
        model = torch.nn.Linear(2, 1)
        good = model.state_dict()
        strict_base_state(model, good)
        with self.assertRaises(RuntimeError):
            strict_base_state(model, {k: v for k, v in good.items() if k != 'bias'})
        with self.assertRaises(RuntimeError):
            strict_base_state(model, dict(good, unrelated=torch.tensor(1)))
        cfg = make_config(load_suite('formal_kvasir_v1'), 'paired_aug_seed666', True)
        cfg.init_from = 'old_trained_checkpoint.pth'
        with self.assertRaises(ValueError):
            make_signature(cfg)

    def test_actual_pause_resume_all_arms(self):
        namespace = old.make_namespace()
        logger = namespace['logger_config']
        parsed = ast.parse((ROOT/'train_formal.py').read_text(encoding='utf-8'))
        funcs = [n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=funcs, type_ignores=[]), 'train_formal production functions', 'exec'), namespace)
        namespace['logger_config'] = logger
        original_cfg = old.make_cfg
        class Data(old.ToyDataset):
            def __getitem__(self, index):
                batch = super().__getitem__(index)
                batch['text_prompt'] = 'one polyp, located in top left of the image'
                return batch
        def data(cfg, split, records):
            return Data(split.lower(), 'toy', [], old.ToyTransform(augment=split == 'Train'))
        for variant in VARIANTS:
            def config():
                cfg = original_cfg()
                cfg.MODEL.VARIANT = variant
                cfg.TRAIN.DICE_MODE = 'image'
                cfg.TRAIN.NEW_LEARNING_RATE = .003
                cfg.TRAIN.MIN_LEARNING_RATE = .0001
                cfg.RESEARCH = SimpleNamespace(BASE_WEIGHTS='toy', RUN_ID=variant+'_seed666')
                return cfg
            with self.subTest(variant=variant), patch.object(old, 'make_cfg', config), \
                 patch('research_formal.model.build_model', lambda cfg: old.ToyModel()), \
                 patch('research_formal.common.make_signature', lambda cfg: {'digest': 'formal-toy'}), \
                 patch('research_formal.data.make_dataset', data), tempfile.TemporaryDirectory() as directory:
                old.test_repeated_resume(namespace, Path(directory))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
