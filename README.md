# ZoomCLIPSeg

MedCLIPSeg-based model training, detail-branch experiments, SCZ inference, evaluation, and paper comparison protocols.

## Layout

- `MedCLIPSeg/`: requested research packages, training entry points, original model utilities, and their local source dependencies.
- `MedCLIPSeg/**/suites/*.yaml`: original experiment protocols, preserved without modification.
- `tools/launch_*.py` and root `*.cmd`: Windows experiment launchers and their local helpers.
- `UPLOAD_MANIFEST.json`: source-relative paths, SHA-256 hashes, sizes, and inclusion reasons for every copied source file.

The source snapshot preserves the original `MedCLIPSeg/` and workspace-level `tools/` layout. Local Python dependencies referenced by the requested code and launchers are also included. All copied files retain their original bytes.

## Running

Prepare a Python environment at `.venv/` to use the supplied `.cmd` launchers. See `MedCLIPSeg/README.md`, `MedCLIPSeg/requirements.txt`, and the Windows requirements files for the original environment requirements. Each launcher exposes its own actions; for example:

```bat
ResearchDetailFormal.cmd -Action Status
ZoomCompare.cmd -Action Status
```

Datasets, model checkpoints, caches, execution-control state, logs, experiment outputs, and server backups are not part of this code snapshot. Set up the expected local data and model paths before running training or evaluation. Some scripts retain paths and assumptions from the original experiment workspace and may need configuration for a different machine. This upload does not claim a fresh training or evaluation run.

## Upstream and license

Based on [HealthX-Lab/MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg). The original Apache-2.0 license is retained at `MedCLIPSeg/LICENSE`. Upstream source files include local research modifications; compare this snapshot with upstream to inspect those changes. Additional notices bundled with individual components are retained alongside their source.
