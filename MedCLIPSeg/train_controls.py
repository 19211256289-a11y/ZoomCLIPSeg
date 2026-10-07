"""Entry point for the control experiments. Runs the accepted trainer (train_detail_formal.py) unchanged.

Depending on the arm, the accepted trainer is given a single-view objective and/or a dataset variant from
research_controls.recipes. The reference arms (recipe 'paired', data 'standard') replace nothing.
"""
from local_runtime import configure_local_models
configure_local_models()
import sys

import yaml


def main():
    config = sys.argv[sys.argv.index('--config-file')+1]
    with open(config, encoding='utf-8') as stream:
        research = yaml.safe_load(stream)['RESEARCH']
    prompt, recipe, data_mode = research['GENERIC_PROMPT'], research['CONTROL_RECIPE'], research['CONTROL_DATA']
    print('FIXED CLASS SENTENCE:', prompt, '| ARM:', research['CONTROL_ARM'], '| RECIPE:', recipe, '| DATA:', data_mode, flush=True)
    import research_detail_formal.data as data
    import research_detail_formal.objective as objective
    from research_controls import recipes
    data.GENERIC = prompt
    objective.GENERIC = prompt
    if recipe in ('single', 'single_flip'):
        objective.paired_objective = recipes.single_view_objective
    elif recipe != 'paired':
        raise ValueError(recipe)
    if data_mode in ('scale_aug', 'upsampled_detail'):
        data.make_dataset = recipes.make_dataset_factory(data_mode)
    elif data_mode != 'standard':
        raise ValueError(data_mode)
    import train_detail_formal
    return train_detail_formal.main()


if __name__ == '__main__':
    sys.exit(main())
