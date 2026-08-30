import torch
import pytest

from regnav import streaming
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


def test_cached_scene_inference_refreshes_early_while_turning():
    runner = CachedSceneInference(
        ValueModel(), scene_refresh_interval=5, turn_rate_threshold=0.5
    )

    outputs = []
    for value in range(1, 9):
        outputs.append(
            runner(
                {
                    "image": torch.tensor([value]),
                    "ego": torch.tensor([[0.0, float(value == 3), 0.0, 0.0]]),
                    "route_goal": torch.empty(1, 0),
                }
            ).item()
        )

    assert outputs == [1, 1, 3, 3, 3, 3, 3, 8]


def test_cached_scene_inference_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="must be positive"):
        CachedSceneInference(ValueModel(), scene_refresh_interval=0)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_cached_scene_inference_rejects_invalid_turn_rate_threshold(value):
    with pytest.raises(ValueError, match="turn_rate_threshold"):
        CachedSceneInference(
            ValueModel(), scene_refresh_interval=5, turn_rate_threshold=value
        )


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


class TrajectoryModel(ValueModel):
    def decode_scene(self, scene, ego, route_goal):
        trajectory = torch.zeros(len(scene), 1, 3)
        trajectory[..., 0] = scene.reshape(-1, 1)
        return {"trajectory": trajectory}


class LoggingTrajectoryModel(TrajectoryModel):
    def __init__(self):
        super().__init__()
        self.encoded_images = []
        self.decoded_scenes = []

    def encode_scene(self, image):
        self.encoded_images.append(float(image.item()))
        return image

    def decode_scene(self, scene, ego, route_goal):
        self.decoded_scenes.append(float(scene.item()))
        return super().decode_scene(scene, ego, route_goal)


def test_cache_interval_evaluation_resets_scene_between_trajectories():
    samples = []
    for trajectory_id, values in (("a", (1, 2, 3)), ("b", (10, 11))):
        for value in values:
            samples.append(
                (
                    trajectory_id,
                    {
                        "image": torch.tensor([[float(value)]]),
                        "ego": torch.tensor([[0.0, float(value in (2, 11)), 0.0, 0.0]]),
                        "route_goal": torch.empty(1, 0),
                        "target_trajectory": torch.zeros(1, 1, 3),
                        "valid_mask": torch.ones(1, 1, dtype=torch.bool),
                    },
                )
            )

    report = streaming.evaluate_cache_intervals(
        TrajectoryModel(),
        samples,
        scene_refresh_intervals=(2,),
        trajectory_interval=0.5,
        device=torch.device("cpu"),
        adaptive_scene_refresh_interval=2,
    )

    assert report[1]["delta_to_interval_1"]["ade"] == pytest.approx(0.0)
    assert report[2]["delta_to_interval_1"]["ade"] == pytest.approx(0.4)
    assert report[2]["target"]["ade"] == pytest.approx(5.0)
    assert report[2]["refresh_rate"] == pytest.approx(0.6)
    assert report[2]["latency_ms"]["mean"] >= 0
    assert report[2]["turning"]["samples"] == 2
    assert report[2]["turning"]["delta_to_interval_1"]["ade"] == pytest.approx(1.0)
    assert report[2]["turning"]["target"]["ade"] == pytest.approx(5.5)
    assert report["adaptive_2"]["delta_to_interval_1"]["ade"] == pytest.approx(0.2)
    assert report["adaptive_2"]["refresh_rate"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"adaptive_scene_refresh_interval": 0}, "adaptive_scene_refresh_interval"),
        ({"turn_rate_threshold": float("inf")}, "turn_rate_threshold"),
    ],
)
def test_cache_interval_evaluation_rejects_invalid_adaptive_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        streaming.evaluate_cache_intervals(
            TrajectoryModel(),
            (),
            scene_refresh_intervals=(1,),
            trajectory_interval=0.5,
            device=torch.device("cpu"),
            **kwargs,
        )


def _evaluation_samples(values):
    return [
        (
            "route",
            {
                "image": torch.tensor([[float(value)]]),
                "ego": torch.zeros(1, 4),
                "route_goal": torch.empty(1, 0),
                "target_trajectory": torch.zeros(1, 1, 3),
                "valid_mask": torch.ones(1, 1, dtype=torch.bool),
            },
        )
        for value in values
    ]


def test_cache_interval_evaluation_warms_up_first_batch_outside_measurement():
    model = LoggingTrajectoryModel()

    streaming.evaluate_cache_intervals(
        model,
        _evaluation_samples((1,)),
        scene_refresh_intervals=(1,),
        trajectory_interval=0.5,
        device=torch.device("cpu"),
    )

    assert model.encoded_images == [1.0, 1.0]


def test_cache_interval_evaluation_rotates_interval_order_each_frame():
    model = LoggingTrajectoryModel()

    streaming.evaluate_cache_intervals(
        model,
        _evaluation_samples((1, 2, 3)),
        scene_refresh_intervals=(1, 2, 3),
        trajectory_interval=0.5,
        device=torch.device("cpu"),
    )

    assert model.decoded_scenes[3:] == [
        1.0, 1.0, 1.0,
        1.0, 1.0, 2.0,
        1.0, 3.0, 3.0,
    ]
