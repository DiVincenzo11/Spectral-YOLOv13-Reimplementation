# Third-party source and attribution

## YOLOv13 / Ultralytics

- Repository: https://github.com/iMoonLab/yolov13
- Commit: `73289949533efac82bb5f72ec19b746618656bd2`
- Copied path: `ultralytics/` to `vendor/yolov13/ultralytics/`, preserving source headers and configuration files.
- License: AGPL-3.0, retained in `vendor/yolov13/LICENSE` and root `LICENSE`.
- Upstream README retained as `vendor/yolov13/UPSTREAM_README.md`.
- Backbone layers, official regression branches, task-aligned assigner, box geometry, and detection augmentation utilities are reused. New WIO/CP/Mamba integration is under `spectral_yolo/`.

## Mamba

- Reference: https://github.com/state-spaces/mamba/tree/v1.0.1
- Source consulted: `mamba_ssm/modules/mamba_simple.py` and `mamba_ssm/ops/selective_scan_interface.py`.
- License of reference implementation: Apache-2.0.
- `spectral_yolo/mamba.py` is a portable reconstruction of the published Mamba-1 algebra, not a recovered Bio-Mamba source file. CUDA kernel use is optional and supplied by separately installed mamba-ssm.
- The official Mamba license is retained in `vendor/MAMBA_LICENSE`.

## Wavelets

- pytorch_wavelets: https://github.com/fbcotter/pytorch_wavelets (BSD-3-Clause), installed as dependency, not vendored.
- PyWavelets: https://github.com/PyWavelets/pywt (MIT), installed as dependency.

## Paper

Yang et al. Spectral-YOLOv13. Sensors 2026, 26(10), 3265. https://doi.org/10.3390/s26103265

The paper is CC BY. Its PDF and sample photographs are not copied into this code package. All reconstruction ambiguities are recorded in `docs/IMPLEMENTATION.md`.
