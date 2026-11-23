from __future__ import annotations

import hashlib

import numpy as np
import torch

from neordrp_ml_v1.config import ModelConfig
from neordrp_ml_v1.inference import NeoRdRpV1Predictor, read_fasta
from neordrp_ml_v1.model import NeoRdRpV1


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fasta_reader(tmp_path):
    path = tmp_path / "input.faa"
    path.write_text(">a description\nACD\n>b\nEFG\n", encoding="utf-8")
    assert read_fasta(path) == [("a", "ACD"), ("b", "EFG")]


def test_tiny_checkpoint_round_trip(tmp_path):
    config = ModelConfig(
        d_model=64,
        n_layers=1,
        d_state=16,
        d_head=16,
        chunk_size=8,
        dropout=0.0,
        gradient_checkpointing=False,
        esm_dim=32,
        protrek_dim=48,
        fusion_hidden=48,
        boundary_head_hidden=64,
    )
    model = NeoRdRpV1(config, 2)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "config": {"model": config.__dict__},
            "model_state": model.state_dict(),
            "family_names": ["FamilyA", "FamilyB"],
            "calibration": {
                "detection_threshold_fpr_1e_3": 0.5,
                "family_temperature": 1.0,
                "family_abstain_threshold": 0.1,
                "start_decode_method": "argmax",
                "end_decode_method": "argmax",
                "start_bias": 0.0,
                "end_bias": 0.0,
                "start_low": -2.0,
                "start_high": 2.0,
                "end_low": -2.0,
                "end_high": 2.0,
            },
        },
        checkpoint,
    )
    features = tmp_path / "features.npz"
    rng = np.random.default_rng(7)
    np.savez_compressed(
        features,
        protein_ids=np.asarray(["protein_1"]),
        esm=rng.normal(size=(1, 32)).astype(np.float16),
        esm_mask=np.asarray([True]),
        protrek=rng.normal(size=(1, 48)).astype(np.float16),
        protrek_mask=np.asarray([True]),
    )
    predictor = NeoRdRpV1Predictor(
        checkpoint=checkpoint,
        device="cpu",
        expected_sha256=_sha256(checkpoint),
    )
    prediction = predictor.predict([("protein_1", "ACDEFGHIK")], features)[0]
    assert prediction.protein_id == "protein_1"
    assert 1 <= prediction.qstart < prediction.qend <= 9
    assert prediction.checkpoint_sha256 == _sha256(checkpoint)
