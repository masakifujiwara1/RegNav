from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch
import pytest

from regnav import benchmark
from regnav.benchmark import _image_shape
from regnav.config import ModelConfig


def test_image_shape_includes_temporal_context():
    config = ModelConfig(image_size=(28, 42), context_frames=4)

    assert _image_shape(config) == (1, 3, 4, 42, 28)


class SplitModel(torch.nn.Module):
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


class CountingSplitModel(SplitModel):
    def __init__(self):
        super().__init__()
        self.encoder_calls = 0

    def encode_scene(self, image):
        self.encoder_calls += 1
        return super().encode_scene(image)


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


def test_benchmark_reports_streaming_latency_when_refresh_interval_is_set():
    batch = {
        "image": torch.ones(1, 3, 2, 2),
        "ego": torch.zeros(1, 4),
        "route_goal": torch.zeros(1, 6),
    }

    result = benchmark._benchmark_stages(
        SplitModel(), batch, torch.device("cpu"), iterations=2, warmup=0,
        scene_refresh_interval=2,
    )

    assert "streaming" in result


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



def test_benchmark_parser_accepts_real_image_replay_inputs():
    args = benchmark.build_parser().parse_args(
        [
            "--checkpoint",
            "best.pt",
            "--device",
            "cpu",
            "--manifest",
            "manifest.jsonl",
            "--split",
            "test",
            "--samples",
            "100",
            "--output",
            "replay.json",
        ]
    )

    assert args.manifest == Path("manifest.jsonl")
    assert args.samples == 100

def test_real_image_replay_reports_adaptive_refresh_and_stage_latency(tmp_path):
    paths = []
    for value in range(4):
        path = tmp_path / f"{value}.jpg"
        Image.new("RGB", (8, 8), (value, value, value)).save(path)
        paths.append(path)

    frames = []
    for index, path in enumerate(paths):
        frames.append(
            (
                (path,),
                torch.tensor([0.0, float(index == 1), 0.0, 0.0]),
                torch.zeros(6),
            )
        )

    model = CountingSplitModel()
    result = benchmark._benchmark_replay(
        model,
        frames,
        image_size=(8, 8),
        device=torch.device("cpu"),
        scene_refresh_interval=3,
        turn_rate_threshold=0.5,
    )

    assert result["samples"] == 4
    assert result["scene_refreshes"] == 2
    assert result["refresh_rate"] == 0.5
    assert result["scene_refresh_interval"] == 3
    assert result["turn_rate_threshold"] == 0.5
    assert result["stages"].keys() == {
        "image_preprocess",
        "input_transfer",
        "scene_encoder",
        "cached_head",
        "streaming_model",
        "end_to_end",
    }
    assert result["warmup_frames"] == 1
    assert result["image_io"] == "warm_os_cache"
    assert model.encoder_calls == 3


def test_replay_frames_stay_within_one_trajectory(tmp_path):
    class ReplayDataset:
        config = SimpleNamespace(context_frames=1)
        index = [(0, 0), (0, 1), (1, 0)]

        def __getitem__(self, index):
            return {
                "ego": torch.full((4,), float(index)),
                "route_goal": torch.zeros(6),
            }

    records = [
        SimpleNamespace(path=tmp_path / "a", trajectory_id="a"),
        SimpleNamespace(path=tmp_path / "b", trajectory_id="b"),
    ]
    frames, trajectory_id = benchmark._replay_frames(
        ReplayDataset(), records, samples=2
    )

    assert trajectory_id == "a"
    assert len(frames) == 2
    assert frames[0][0] == (tmp_path / "a" / "0.jpg",)


def test_real_image_replay_rejects_non_cpu_device(tmp_path):
    path = tmp_path / "0.jpg"
    Image.new("RGB", (8, 8)).save(path)
    frames = [((path,), torch.zeros(4), torch.zeros(6))]

    with pytest.raises(ValueError, match="CPU"):
        benchmark._benchmark_replay(
            SplitModel(),
            frames,
            image_size=(8, 8),
            device=torch.device("cuda"),
            scene_refresh_interval=3,
            turn_rate_threshold=0.5,
        )


def test_benchmark_parser_rejects_more_than_100_replay_samples():
    with pytest.raises(SystemExit):
        benchmark.build_parser().parse_args(["--checkpoint", "best.pt", "--samples", "101"])
