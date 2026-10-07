"""Use the installed model cache without contacting Hugging Face during a run."""
import os
from pathlib import Path


def configure_local_models():
    # Must run before importing transformers/huggingface_hub: they cache these flags.
    workspace = Path(__file__).resolve().parent.parent
    os.environ.setdefault('HF_HOME', str(workspace / '.cache' / 'huggingface'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
