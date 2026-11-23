# Reproducibility

## Inference verification

1. Install the wheel.
2. Set `HF_TOKEN` for the private Hugging Face repository.
3. Run `neordrp-ml-v1 demo --output demo_prediction.json`.
4. Verify that the checkpoint SHA-256 equals
   `3c6bbe670a53824a8f2645936d0ca44dbec374ed4ee4f9b5c94c8f606e374c97`.
5. Verify that the output contains one row for `E1:L0/Q2N379`, a 1-based
   interval, and the same checkpoint hash.

The bundled feature bundle contains only the demonstration protein. It is not
training data.

## Training verification

Run:

```bash
bash scripts/reproduce_v1.sh
```

The script installs training dependencies, downloads and hashes the processed
bundle, checks for CUDA, runs both stages, and writes immutable configuration
snapshots and per-epoch checkpoints.

For a transient smoke test:

```bash
python scripts/reproduce_v1.py --limit-steps 1
```

The smoke test is still GPU-only and should not be reported as a reproduced
training run.

## Determinism

The trainer restores Python, NumPy, CPU PyTorch, CUDA, and data-sampling random
state when resuming. Exact bitwise identity is not guaranteed across GPU models,
CUDA versions, PyTorch versions, or parallel kernels. Report the hardware and
software environment with reproduced results.
