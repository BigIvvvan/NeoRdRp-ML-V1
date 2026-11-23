# NeoRdRp-ML V1

[日本語](README.ja.md) | [Model and data](https://huggingface.co/BigIvvvan/NeoRdRp-ML-V1)

NeoRdRp-ML V1 identifies RNA-dependent RNA polymerases (RdRps), predicts the
likely viral family, and returns a residue interval for the polymerase module.
This repository contains the training code, inference code, reproduction
scripts, tests, and a bundled single-protein inference demonstration.

The model combines a bidirectional Mamba encoder over amino-acid tokens with
pooled ESM-C and ProTrek representations. ESM-C and ProTrek are used both as
input context and as training-time distillation targets. The final V1 model
has 17,090,558 parameters.

## Outputs

For each protein, the package reports:

- an RdRp probability and a calibrated binary call;
- the most likely viral family, its probability, and an abstention flag;
- a 1-based closed `Qstart`/`Qend` interval and calibrated uncertainty bounds.

Coordinates are 1-based and inclusive. When a protein is longer than the
4,096-residue inference window, the result includes the evaluated center window
so that coordinates remain auditable.

## Install

Python 3.11 or later is required.

```bash
python -m pip install neordrp-ml-v1
```

The wheel is also attached to the GitHub release:

```bash
python -m pip install neordrp_ml_v1-1.0.0-py3-none-any.whl
```

The model repository is private. Set a Hugging Face access token before the
first run unless `--checkpoint` points to a local V1 checkpoint:

```bash
export HF_TOKEN=your_token
```

## Single-protein demonstration

The wheel contains a real FASTA record and its precomputed ESM-C/ProTrek
features. The model checkpoint is downloaded and SHA-256 verified on first use.

```bash
neordrp-ml-v1 demo
```

Expected output from the bundled CPU run:

```text
protein_id       rdrp_probability  rdrp_called  family           family_probability  family_abstained  qstart  qend
E1:L0/Q2N379     1.0               true         Caliciviridae   0.9936              false             6       481
```

To keep the complete machine-readable result:

```bash
neordrp-ml-v1 demo --output demo_prediction.json
```

## Predict a custom FASTA

Install the optional embedding dependencies and compute 2,048-residue pooled
features:

```bash
python -m pip install "neordrp-ml-v1[embed]"
neordrp-ml-v1 embed --fasta proteins.faa --output proteins_features.npz --device cuda
neordrp-ml-v1 predict --fasta proteins.faa --features proteins_features.npz --output predictions.json
```

The feature extractor expects access to `esmc_600m` and
`westlake-repl/ProTrek_650M`. Set `NEORDRP_PROTREK_DIR` if ProTrek weights are
stored outside the Hugging Face cache. The released NeoRdRp-ML checkpoint is
downloaded separately and does not include those upstream model weights.

## Reproduce training

Training is GPU-only. The one-command Linux workflow downloads the processed
training bundle from Hugging Face, verifies its SHA-256, and runs the two V1
stages:

```bash
bash scripts/reproduce_v1.sh
```

Equivalent Python entry point:

```bash
python -m pip install -e ".[train]"
python scripts/reproduce_v1.py
```

Stage 1 trains detection, 65-family classification, boundary prediction, and
feature distillation. Stage 2 initializes from stage 1 and emphasizes boundary
quality. Configurations are in `configs/`. Processed training data are
approximately 826 MB compressed and about 879 MB uncompressed.

See [Training](docs/training.md), [Data card](docs/data_card.md),
[Model card](docs/model_card.md), and
[Reproducibility](docs/reproducibility.md) for details.

## Scientific status

V1 is a research release, not a clinical or diagnostic tool. On the frozen
RefSeq detection benchmark, V1 had AUROC 0.7621 and recalled 181 of 1,235
positives at no more than 60 false positives among 60,000 negatives. These
results exceeded the included HMM baselines, but Rider and an earlier internal
model ranked higher in this exposed benchmark.

Family classification was comparatively strong and well calibrated in the
closed 65-family evaluation. Boundary localization remains the principal V1
limitation: only 1 of 41 reference proteins had the start within 10 residues,
3 of 41 had the end within 10 residues, and no protein met both criteria.
Do not describe V1 localization as successful.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
python -m build
```

The package and released model are distributed under the MIT License. Upstream
ESM-C and ProTrek weights remain subject to their own licenses.
