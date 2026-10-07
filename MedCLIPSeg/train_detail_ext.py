"""Entry point for the extension suites. Runs the accepted trainer (train_detail_formal.py) unchanged.

The accepted data and objective modules contain the fixed class sentence as a module constant.
This wrapper sets that constant from the run's config (RESEARCH.GENERIC_PROMPT) before training, so a
non-polyp dataset uses its own fixed sentence. For Kvasir the value is identical to the accepted one.
"""
from local_runtime import configure_local_models
configure_local_models()
import sys

import yaml


def main():
    config = sys.argv[sys.argv.index('--config-file')+1]
    with open(config, encoding='utf-8') as stream:
        prompt = yaml.safe_load(stream)['RESEARCH']['GENERIC_PROMPT']
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('RESEARCH.GENERIC_PROMPT is required')
    print('FIXED CLASS SENTENCE FOR THIS RUN:', prompt, flush=True)
    import research_detail_formal.data as data
    import research_detail_formal.objective as objective
    data.GENERIC = prompt
    objective.GENERIC = prompt
    import train_detail_formal
    return train_detail_formal.main()


if __name__ == '__main__':
    sys.exit(main())
