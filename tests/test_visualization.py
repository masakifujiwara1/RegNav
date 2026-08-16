import json

import torch
from PIL import Image

from regnav.visualization import (
    dataset_labels,
    render_trajectory_png,
    write_visualization_index,
)


def _sample() -> dict[str, object]:
    return {
        "dataset": "recon",
        "dataset_family": "vint",
        "subset": "recon",
        "robot": "jackal",
        "trajectory_id": "route-a",
        "frame": 3,
        "metrics": {
            "model": {"ade": 0.1, "fde": 0.2, "heading_error": 0.3},
        },
        "image": torch.zeros(3, 12, 16),
        "target": torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.5, 0.1]]),
        "prediction": torch.tensor([[1.0, 0.1, 0.0], [1.8, 0.4, 0.1]]),
        "constant_velocity": torch.tensor([[0.8, 0.0, 0.0], [1.6, 0.0, 0.0]]),
        "route_only": torch.tensor([[0.5, 0.1, 0.0], [1.0, 0.2, 0.0]]),
        "obstacles": torch.tensor([[1.0, 0.8]]),
    }


def test_dataset_labels_default_to_vint_family():
    assert dataset_labels("recon") == ("vint", "recon")
    assert dataset_labels("vint/recon") == ("vint", "recon")


def test_render_trajectory_png_contains_input_and_top_down_plot(tmp_path):
    output = tmp_path / "sample.png"

    render_trajectory_png(_sample(), output)

    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.width >= 900
        assert image.height >= 500


def test_write_visualization_index_records_metadata_and_metrics(tmp_path):
    sample = _sample()
    image_path = tmp_path / "0001-route-a.png"
    render_trajectory_png(sample, image_path)

    rows = write_visualization_index(tmp_path, [sample], [image_path.name])

    assert rows[0]["file"] == image_path.name
    assert rows[0]["dataset"] == "recon"
    assert rows[0]["subset"] == "recon"
    index = (tmp_path / "index.html").read_text()
    assert "route-a" in index
    assert "ADE" in index
    saved = json.loads((tmp_path / "samples.jsonl").read_text().splitlines()[0])
    assert saved["dataset_family"] == "vint"
    assert saved["robot"] == "jackal"
    assert saved["metrics"]["model"]["fde"] == 0.2
