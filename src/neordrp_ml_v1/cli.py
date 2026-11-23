"""NeoRdRp-ML V1 command-line interface."""

from __future__ import annotations

import argparse
import importlib.resources
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .inference import (
    DEFAULT_MODEL_FILE,
    DEFAULT_REPO_ID,
    NeoRdRpV1Predictor,
    predictions_to_json,
    read_fasta,
    write_json,
)


def _manifest_hash() -> str | None:
    resource = importlib.resources.files("neordrp_ml_v1").joinpath(
        "assets/model_manifest.json"
    )
    if not resource.is_file():
        return None
    return str(json.loads(resource.read_text(encoding="utf-8"))["sha256"])


def _print_predictions(predictions) -> None:
    print(
        "protein_id\trdrp_probability\trdrp_called\tfamily\tfamily_probability\t"
        "family_abstained\tqstart\tqend"
    )
    for row in predictions:
        print(
            f"{row.protein_id}\t{row.rdrp_probability:.8g}\t{int(row.rdrp_called)}\t"
            f"{row.family}\t{row.family_probability:.8g}\t"
            f"{int(row.family_abstained)}\t{row.qstart}\t{row.qend}"
        )


def _predictor(args) -> NeoRdRpV1Predictor:
    return NeoRdRpV1Predictor(
        checkpoint=args.checkpoint,
        device=args.device,
        repo_id=args.repo_id,
        model_file=args.model_file,
        expected_sha256=None if args.no_hash_check else _manifest_hash(),
        max_length=args.max_length,
    )


def _predict(args) -> int:
    predictor = _predictor(args)
    predictions = predictor.predict_fasta(args.fasta, args.features)
    _print_predictions(predictions)
    if args.output:
        write_json(
            predictions_to_json(predictions, predictor.checkpoint_path, args.features),
            args.output,
        )
    return 0


def _demo(args) -> int:
    assets = importlib.resources.files("neordrp_ml_v1").joinpath("assets")
    fasta = Path(str(assets.joinpath("demo.fasta")))
    features = Path(str(assets.joinpath("demo_features.npz")))
    predictor = _predictor(args)
    predictions = predictor.predict_fasta(fasta, features)
    _print_predictions(predictions)
    if args.output:
        write_json(
            predictions_to_json(predictions, predictor.checkpoint_path, features),
            args.output,
        )
    return 0


def _embed(args) -> int:
    from .embedding import ProteinBackbone, embed_sequences

    records = read_fasta(args.fasta)
    ids = [identifier for identifier, _ in records]
    sequences = [sequence for _, sequence in records]
    import torch

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    esm_model = ProteinBackbone("esmc_600m", args.device, torch_dtype)
    esm = embed_sequences(esm_model, sequences, args.max_tokens, args.batch_size)
    del esm_model
    protrek_model = ProteinBackbone("protrek_650m", args.device, torch_dtype)
    protrek = embed_sequences(
        protrek_model, sequences, args.max_tokens, args.batch_size
    )
    np.savez_compressed(
        args.output,
        protein_ids=np.asarray(ids),
        esm=esm.astype(np.float16),
        esm_mask=np.ones(len(ids), dtype=bool),
        protrek=protrek.astype(np.float16),
        protrek_mask=np.ones(len(ids), dtype=bool),
    )
    print(f"wrote {len(ids)} feature rows to {args.output}")
    return 0


def _model_info(args) -> int:
    predictor = _predictor(args)
    print(
        json.dumps(
            {
                "checkpoint": str(predictor.checkpoint_path),
                "checkpoint_sha256": predictor.checkpoint_sha256,
                "families": predictor.family_names,
                "model_config": predictor.model_config.__dict__,
                "calibration": predictor.calibration,
            },
            indent=2,
        )
    )
    return 0


def _add_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", help="Local V1 checkpoint; downloads when omitted")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--no-hash-check", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neordrp-ml-v1",
        description=(
            "Detect RNA-dependent RNA polymerases, classify their viral family, "
            "and predict the catalytic interval."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the bundled offline inference demo")
    _add_checkpoint_arguments(demo)
    demo.add_argument("--output", help="Optional JSON result path")
    demo.set_defaults(handler=_demo)

    predict = subparsers.add_parser(
        "predict", help="Predict proteins using FASTA plus a precomputed feature bundle"
    )
    predict.add_argument("--fasta", required=True)
    predict.add_argument("--features", required=True)
    predict.add_argument("--output", help="Optional JSON result path")
    _add_checkpoint_arguments(predict)
    predict.set_defaults(handler=_predict)

    embed = subparsers.add_parser(
        "embed", help="Compute ESM-C and ProTrek features from a protein FASTA"
    )
    embed.add_argument("--fasta", required=True)
    embed.add_argument("--output", required=True)
    embed.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    embed.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    embed.add_argument("--batch-size", type=int, default=16)
    embed.add_argument("--max-tokens", type=int, default=32768)
    embed.set_defaults(handler=_embed)

    info = subparsers.add_parser("model-info", help="Show the loaded checkpoint manifest")
    _add_checkpoint_arguments(info)
    info.set_defaults(handler=_model_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "max_length", 1) < 1:
        parser.error("--max-length must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
