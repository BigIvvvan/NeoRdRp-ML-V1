from pathlib import Path

from neordrp_ml_v1.config import load_config


def test_released_training_configs_are_gpu_only():
    root = Path(__file__).resolve().parents[1]
    stage1 = load_config(root / "configs/train_stage1.yaml")
    stage2 = load_config(root / "configs/train_stage2.yaml")
    assert stage1.train.device == "cuda"
    assert stage2.train.device == "cuda"
    assert stage1.data.embedding_window == 2048
    assert stage2.data.boundary_sigma_aa == 5.0
    assert stage1.hash != stage2.hash
