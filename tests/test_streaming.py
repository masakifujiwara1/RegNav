import torch
import pytest

from regnav.streaming import CachedSceneInference


class ValueModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def encode_scene(self, image):
        return image

    def decode_scene(self, scene, ego, route_goal):
        return scene


def test_cached_scene_inference_refreshes_at_configured_interval():
    runner = CachedSceneInference(ValueModel(), scene_refresh_interval=3)

    outputs = [
        runner(
            {
                "image": torch.tensor([value]),
                "ego": torch.empty(1, 0),
                "route_goal": torch.empty(1, 0),
            }
        ).item()
        for value in range(1, 7)
    ]

    assert outputs == [1, 1, 1, 4, 4, 4]


def test_cached_scene_inference_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="must be positive"):
        CachedSceneInference(ValueModel(), scene_refresh_interval=0)


class TrainingAwareModel(ValueModel):
    def decode_scene(self, scene, ego, route_goal):
        return torch.tensor(self.training)


def test_cached_scene_inference_uses_eval_mode():
    runner = CachedSceneInference(TrainingAwareModel(), scene_refresh_interval=1)
    output = runner(
        {
            "image": torch.ones(1),
            "ego": torch.empty(1, 0),
            "route_goal": torch.empty(1, 0),
        }
    )

    assert output.item() is False
