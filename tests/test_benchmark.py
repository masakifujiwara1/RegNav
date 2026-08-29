import torch

from regnav import benchmark
from regnav.benchmark import _image_shape
from regnav.config import ModelConfig


def test_image_shape_includes_temporal_context():
    config = ModelConfig(image_size=(28, 42), context_frames=4)

    assert _image_shape(config) == (1, 3, 4, 42, 28)


class SplitModel:
    def encode_scene(self, image):
        return image.mean(dim=(-2, -1))

    def decode_scene(self, scene, ego, route_goal):
        return scene + ego[:, :3] + route_goal[:, :3]

    def __call__(self, batch):
        return self.decode_scene(
            self.encode_scene(batch["image"]),
            batch["ego"],
            batch["route_goal"],
        )


def test_benchmark_reports_end_to_end_encoder_and_cached_head():
    batch = {
        "image": torch.ones(1, 3, 2, 2),
        "ego": torch.zeros(1, 4),
        "route_goal": torch.zeros(1, 6),
    }

    result = benchmark._benchmark_stages(
        SplitModel(), batch, torch.device("cpu"), iterations=2, warmup=1
    )

    assert result.keys() == {"end_to_end", "scene_encoder", "cached_head"}
    for stage in result.values():
        assert stage.keys() == {"p50_ms", "p95_ms", "p99_ms"}
        assert all(value >= 0 for value in stage.values())


class EndToEndOnlyModel:
    def __call__(self, batch):
        return batch["image"]


def test_benchmark_keeps_end_to_end_only_models_compatible():
    batch = {"image": torch.ones(1, 3, 2, 2)}

    result = benchmark._benchmark_stages(
        EndToEndOnlyModel(), batch, torch.device("cpu"), iterations=1, warmup=0
    )

    assert result.keys() == {"end_to_end"}


def test_benchmark_rotates_stage_order_between_iterations():
    calls = []
    operations = {
        "first": lambda: calls.append("first"),
        "second": lambda: calls.append("second"),
        "third": lambda: calls.append("third"),
    }

    benchmark._measure_operations(
        operations, torch.device("cpu"), iterations=3, warmup=0
    )

    assert calls == [
        "first", "second", "third",
        "second", "third", "first",
        "third", "first", "second",
    ]


def test_benchmark_resets_cuda_peak_memory_after_warmup(monkeypatch):
    events = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: events.append("synchronize")
    )
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device: events.append("reset")
    )

    benchmark._measure_operations(
        {"stage": lambda: events.append("operation")},
        torch.device("cuda"), iterations=1, warmup=1,
    )

    assert events[:3] == ["operation", "synchronize", "reset"]

