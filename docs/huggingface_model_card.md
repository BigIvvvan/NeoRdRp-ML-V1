---
license: mit
library_name: pytorch
tags:
  - bioinformatics
  - RNA-virus
  - RNA-dependent-RNA-polymerase
  - protein-classification
  - Mamba
---

# NeoRdRp-ML V1

NeoRdRp-ML V1 predicts whether a protein is an RNA-dependent RNA polymerase
(RdRp), assigns one of 65 viral families, and estimates a 1-based closed
`Qstart`/`Qend` interval.

## Files

- `models/neordrp_ml_v1.pt`: inference-only checkpoint;
- `models/manifest.json`: model size, hashes, and source checkpoint identity;
- `data/neordrp_ml_v1_training_bundle.tar.gz`: processed V1 training tables and
  frozen ESM-C/ProTrek feature caches;
- `data/neordrp_ml_v1_training_bundle.tar.gz.sha256`: bundle checksum.

The model checkpoint SHA-256 is
`3c6bbe670a53824a8f2645936d0ca44dbec374ed4ee4f9b5c94c8f606e374c97`.

The training bundle SHA-256 is
`9a7d22406267bd2ae83b7bd98b8fa0fb257f6c62ae471714efdbb9cb576fd473`.

## Install and run

```bash
python -m pip install "neordrp-ml-v1 @ git+https://github.com/BigIvvvan/NeoRdRp-ML-V1.git"
export HF_TOKEN=your_token
neordrp-ml-v1 demo
```

The GitHub repository contains complete inference code, two-stage training
code, single-command data download, tests, English documentation, and Japanese
documentation.

## Evaluation and limitations

On the exposed RefSeq detection benchmark, V1 achieved AUROC 0.7621 and
recalled 181/1,235 positives with 60/60,000 false positives. V1 outperformed
the included HMM baselines but did not rank above every neural baseline.

The closed 65-family evaluation reached 97.676% top-1 accuracy, 0.9047
macro-F1, and 0.0045 expected calibration error.

Boundary localization is not yet successful. Only 1/41 starts and 3/41 ends
were within 10 residues of the reference, and no protein met both criteria.
Treat the interval as a research estimate, not an experimentally validated
catalytic-domain boundary.

This model is not intended for clinical diagnosis or autonomous biosafety
decisions. Upstream ESM-C and ProTrek weights are not redistributed here and
remain subject to their own licenses.
