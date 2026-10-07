"""Entry point for the native-resolution pilot. Runs the accepted trainer (train_detail_formal.py) unchanged.

For 448 arms the accepted trainer is given this pilot's model builder and dataset (448 mask); the loss,
paired-reflection objective, optimizer, schedule, checkpointing and pause logic are the accepted ones.
For 224 arms nothing is replaced.
"""
from local_runtime import configure_local_models
configure_local_models()
import sys

import yaml


def main():
    config = sys.argv[sys.argv.index('--config-file')+1]
    with open(config, encoding='utf-8') as stream:
        research = yaml.safe_load(stream)['RESEARCH']
    prompt, mode = research['GENERIC_PROMPT'], research['HIRES_MODE']
    if mode not in ('none', 'sup448', 'detail448'):
        raise ValueError(mode)
    print('FIXED CLASS SENTENCE FOR THIS RUN:', prompt, '| RESOLUTION MODE:', mode, flush=True)
    import research_detail_formal.data as data
    import research_detail_formal.objective as objective
    import research_detail_formal.model as model
    data.GENERIC = prompt
    objective.GENERIC = prompt
    if mode != 'none':
        from research_hires.model import build_model
        from research_hires.data import make_dataset
        model.build_model = build_model
        data.make_dataset = make_dataset
    import train_detail_formal
    return train_detail_formal.main()


if __name__ == '__main__':
    sys.exit(main())
