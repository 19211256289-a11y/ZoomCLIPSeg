"""Check every file named in the official prompt splits before a job starts."""
import json
from pathlib import Path

import openpyxl
from utils.main_utils import load_cfg_from_cfg_file

ROOT = Path(__file__).resolve().parent


def dataset_readiness(dataset, percentage=100, training=False):
    cfg = load_cfg_from_cfg_file(str(ROOT / 'configs' / f'{dataset}.yaml'))
    suffix = '' if percentage == 100 else f'_{percentage}'
    splits = [('Test', 'Test_text_original.xlsx', cfg.DATASET.TEST_PATH)]
    if training:
        splits = [('Train', f'Train_text{suffix}.xlsx', cfg.DATASET.TRAIN_PATH),
                  ('Val', f'Val_text{suffix}.xlsx', cfg.DATASET.VAL_PATH)] + splits
    report = {'dataset': dataset, 'percentage': percentage, 'training': training,
              'ready': True, 'splits': {}, 'errors': [], 'warnings': []}
    if dataset == 'EUS':
        marker = ROOT.parent / 'reports' / 'eus_preparation.json'
        try:
            state = json.loads(marker.read_text(encoding='utf-8'))
            if state['status'] == 'blocked':
                report['blocked_reason'] = state.get('error', 'EUS source data require clarification')
            if state['status'] != 'completed':
                raise ValueError('preparation incomplete')
        except (OSError, ValueError, KeyError):
            report['errors'].append('EUS images/annotations require verified preparation (reports/eus_preparation.json).')
    for name, workbook, folder in splits:
        path = ROOT / cfg.DATASET.TEXT_PROMPT_PATH / workbook
        missing = []
        count = 0
        seen = {}
        repeated = []
        try:
            book = openpyxl.load_workbook(path, read_only=True, data_only=True)
            try:
                rows = iter(book.active.values)
                columns = next(rows)
                image_index, mask_index = columns.index('Image'), columns.index('Ground Truth')
                for row in rows:
                    if row[image_index] is None:
                        continue
                    count += 1
                    image_name, mask_name = str(row[image_index]), str(row[mask_index])
                    if image_name in seen:
                        repeated.append(image_name)
                        # The official ISIC Train_text(.xlsx/_50.xlsx) includes
                        # one image with two descriptions and the same mask.
                        # Keep all author training rows; evaluation must be unique.
                        if name == 'Train' and seen[image_name] == mask_name:
                            report['warnings'].append(
                                f'{name}: repeated image {image_name} with the same mask; '
                                'preserving all training prompt rows')
                        else:
                            report['errors'].append(f'{name}: duplicate image {image_name}')
                    seen[image_name] = mask_name
                    for kind, filename in [('img', image_name), ('label', mask_name)]:
                        file = ROOT / folder / kind / filename
                        if not file.is_file() or file.stat().st_size == 0:
                            missing.append(str(file))
            finally:
                book.close()
            if not count:
                report['errors'].append(f'{name}: empty prompt workbook')
        except (OSError, ValueError, StopIteration) as exc:
            report['errors'].append(f'{name}: {exc}')
        report['splits'][name] = {'samples': count, 'unique_images': len(seen),
                                  'repeated_images': repeated, 'missing_count': len(missing),
                                  'missing_examples': missing[:5]}
        if missing:
            report['errors'].append(f'{name}: {len(missing)} missing/empty files')
    report['ready'] = not report['errors']
    return report
