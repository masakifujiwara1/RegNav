import pytest

def test_cache_evaluation_parser_accepts_intervals_and_cpu_device():
    from regnav.cache_evaluate import build_parser

    args = build_parser().parse_args(
        [
            "--checkpoint", "model.pt",
            "--manifest", "manifest.jsonl",
            "--split", "test",
            "--output", "cache-report.json",
            "--scene-refresh-intervals", "1", "2", "5",
            "--adaptive-scene-refresh-interval", "5",
            "--device", "cpu",
            "--turn-rate-threshold", "0.5",
        ]
    )

    assert args.scene_refresh_intervals == [1, 2, 5]
    assert args.adaptive_scene_refresh_interval == 5
    assert args.device == "cpu"
    assert args.turn_rate_threshold == 0.5


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf"])
def test_cache_evaluation_parser_rejects_invalid_turn_rate_threshold(value):
    from regnav.cache_evaluate import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--checkpoint", "model.pt",
                "--manifest", "manifest.jsonl",
                "--split", "test",
                "--output", "cache-report.json",
                "--turn-rate-threshold", value,
            ]
        )
