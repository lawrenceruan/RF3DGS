# Accepted RF3DGS baseline snapshot

Branch: `modified_baseline`. Snapshot date: 9 November 2026.
Parent: `00e7b565828ea3011d01f721a9327b0bbc199306` (original 013 import).

This branch saves the current accepted live RF3DGS source. The original import remains in the parent commit and on `main`. See [BASELINE_DEFAULTS.md](BASELINE_DEFAULTS.md) for the selected defaults and the existing-prior-cache caveat. Its final paragraph describes the latest sampling-default switch, not the full difference from the original import.

## Accepted behavior

- Zero-safe exclusion-loss fourth-root gradients; positive-energy formula retained.
- Repeated identical exclusion computations, unused VGG initialization, and disconnected histogram work removed.
- Complete T/R/beta checkpoints with optimizer, random-state and camera-order restoration; invalid legacy one-layer checkpoints rejected.
- Nonfinite Gaussian checks at save time and recorded training arguments, including warnings for inactive legacy controls.
- Legacy self-prior with historical pixel sampling; original sampled SSIM.
- Depth visualization uses OpenCV. The pseudo-prior folder can be selected explicitly.

Optional prior/loss helpers are preserved because this is a source snapshot; they are not accepted quality improvements and remain disabled by default. Corrected sampling requires `--corrected_pixel_sampling`. Do not pass experimental trainer flags when reproducing this baseline. Fused SSIM, brightness-confidence and contribution-pruning experiment snapshots are not promoted here.

Regenerate priors using the historical default or select a verified historical cache. Changing the generator default does not alter existing priors. Dataset paths, scene splits and training parameters still need to be supplied for the intended scene; existing batch scripts retain their original scene-specific settings.

## Publication checks

The 70 transferred core source/document files were verified against SHA-256 hashes from the live source snapshot. Python compilation and historical-prior/default-off routing checks passed. No new GPU training was run for this publication. This branch makes no new quality or speed claim.

Experiment outputs, datasets, model checkpoints and environments are excluded from this update. Existing upstream/vendor code and licensing are retained from the original import.
