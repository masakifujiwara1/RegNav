from pathlib import Path

import pytest

from regnav.config import ModelConfig, load_yaml_config
from regnav.contracts import RegNavBatch, RegNavOutput


def test_regnav_config_matches_design():
    model, training = load_yaml_config(
        Path("config/model/regnav.yaml"), Path("config/training/default.yaml")
    )

    assert model.image_size == (448, 252)
    assert model.num_poses == 8
    assert model.num_proposals == 8
    assert model.d_model == 256
    assert model.lora_rank == 8
    assert training.lora_start_epoch == 2


def test_invalid_horizon_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("num_poses: 7\ninterval: 0.5\nhorizon: 4.0\n")

    with pytest.raises(ValueError, match=r"num_poses \* interval"):
        ModelConfig.from_yaml(path)


def test_context_frames_must_be_positive():
    with pytest.raises(ValueError):
        ModelConfig(context_frames=0)



def test_tensor_contract_keys_are_stable():
    assert RegNavBatch.__required_keys__ == {
        "image",
        "ego",
        "route_goal",
        "target_trajectory",
        "valid_mask",
    }
    assert RegNavBatch.__optional_keys__ == {
        "obstacle_points",
        "obstacle_points_mask",
    }
    assert RegNavOutput.__required_keys__ == {
        "trajectory",
        "proposals",
        "scores",
        "score_components",
    }
    assert RegNavOutput.__optional_keys__ == {"refinements"}
