# RF3DGS live baseline defaults — 9 November 2026

Accepted core: zero-safe exclusion gradient; redundant exclusion computation removed; unused VGG initialization and disconnected histogram work removed; complete layered checkpoint/RNG restoration and run metadata.

Self-prior: legacy method with HISTORICAL pixel sampling by default. Corrected pixel-center sampling is optional/off and requires `--corrected_pixel_sampling` in warp_images_1st_stage.py. Existing `--legacy_pixel_sampling` remains supported. These flags are mutually exclusive and apply only to prior_mode=legacy.

Training: original sampled SSIM. New experimental prior/loss/module options remain off. This change does not enable fused SSIM, brightness confidence or contribution pruning.

Existing cached pseudo-GT images and immutable experiment snapshots retain their original sampling conventions. To run this baseline, regenerate priors with the live script's default or explicitly use a verified historical-sampling cache. A training run that loads an old corrected cache is still using corrected priors regardless of the new generator default.

Only warp_images_1st_stage.py and this documentation were changed in the live source. The trainer hash was verified unchanged. Backup, patch, routing checks and deployment hashes are saved under runs/baseline_refinement/RF3DGS_default_sampling_20261109.
