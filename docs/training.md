# Training

## Model blocks

1. A 33-token amino-acid vocabulary.
2. A bidirectional, six-layer chunked state-space Mamba encoder.
3. Separate ESM-C and ProTrek projections with a missing-modality-aware gate.
4. Masked mean pooling and a joint detection/family representation.
5. Detection and 65-family classification heads.
6. Shared residue boundary trunk with independent start and end heads.
7. Feature, contrastive, and optional detection-logit knowledge distillation.

ESM-C and ProTrek representations have 1,152 and 1,280 dimensions,
respectively. Features are pooled over 2,048-residue windows with stride 512.

## Two-stage recipe

Stage 1 (`configs/train_stage1.yaml`) trains all three tasks jointly:

- detection: weighted binary cross-entropy;
- family: label-smoothed cross-entropy;
- boundary: Gaussian-smoothed endpoint likelihood on weak labels;
- distillation: projected-feature MSE and symmetric contrastive alignment.

Stage 2 (`configs/train_stage2.yaml`) loads the stage-1 `last.pt` file and
reduces the base learning rate. Detection and family losses remain active while
boundary loss receives greater weight, including an expected-coordinate L1 term.

The development split selects checkpoints. The calibration split determines the
0.1% false-positive detection threshold, family temperature, abstention
threshold, endpoint decoding method, endpoint bias, and 90% endpoint intervals.
Evaluation-only locked data are not present in the release bundle.

## Resource requirements

Training is intentionally restricted to CUDA. A 24 GB GPU can run a smaller
batch through the same model, but the released recipe was designed for a large
GPU using bf16, gradient checkpointing, token-budgeted batching, and gradient
accumulation. Full reproduction is expected to take several hours to roughly a
day depending on GPU throughput and storage. It is not a CPU workload.

## Outputs

Each run writes:

- `config.frozen.yaml`: immutable resolved configuration and hash;
- `metrics.jsonl`: training and development metrics;
- `last.pt`: resumable optimizer, model, and RNG state;
- `best.pt`: best development checkpoint;
- `epoch_XX.pt`: one checkpoint per completed epoch;
- `calibration.json`: calibration-only thresholds;
- `summary.json`: final development and calibration summary.
