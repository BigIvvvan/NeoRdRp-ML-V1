# Model card: NeoRdRp-ML V1

## Model details

- Architecture: bidirectional six-layer Mamba with ESM-C/ProTrek fusion.
- Parameters: 17,090,558.
- Families: 65.
- Input: protein sequence plus pooled ESM-C and ProTrek features.
- Outputs: RdRp probability, family probability, `Qstart`, and `Qend`.
- Coordinate convention: 1-based closed intervals.
- Released checkpoint: `models/neordrp_ml_v1.pt`.
- Released model SHA-256:
  `3c6bbe670a53824a8f2645936d0ca44dbec374ed4ee4f9b5c94c8f606e374c97`.
- Source training checkpoint SHA-256:
  `10b346923f182def33ea88a130afdbb0b9777195bd9cf6a2f0aaf32e2820c1ef`.

The released file is an inference-only serialization. It excludes optimizer
state and random-number-generator state from the training checkpoint.

## Evaluation

The following values are frozen point estimates and should be reported with
their dataset definitions:

- RefSeq detection benchmark: AUROC 0.7621; 181/1,235 positives recalled with
  60/60,000 false positives at the 0.1% false-positive operating point.
- Closed 65-family benchmark: 97.676% top-1 accuracy, 0.9047 macro-F1, and
  0.0045 expected calibration error.
- Structural boundary reference set: 1/41 starts within 10 residues, 3/41 ends
  within 10 residues, and 0/41 proteins with both ends within 10 residues.

The detection and family benchmarks are exposed historical evaluation sets,
not untouched holdouts. Detection ranking was not uniformly superior to all
baselines. The boundary result does not support a claim of accurate complete
interval localization.

## Intended use

Use V1 for research on RdRp detection, family assignment, and the study of
sequence-model boundary uncertainty. Treat `Qstart/Qend` as a model estimate,
not an experimentally measured domain boundary.

## Out-of-scope use

- clinical diagnosis;
- regulatory or biosafety decisions;
- autonomous pathogen discovery without expert review;
- interpreting the predicted interval as an exact catalytic-domain annotation;
- use on nucleotide sequences without translation.
