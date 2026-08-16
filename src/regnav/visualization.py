from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch


def dataset_labels(dataset: str) -> tuple[str, str]:
    """Return ``(family, subset)`` for a manifest dataset label."""
    family, separator, subset = dataset.partition("/")
    return (family, subset) if separator else ("vint", dataset)


def _xy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("trajectory values must have shape (poses, >=2)")
    return array[:, :2]


def _image(value: object) -> Image.Image:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError("image must have shape (height, width, 3) or (3, height, width)")
    if array.shape[0] == 3 and array.shape[2] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3) or (3, height, width)")
    if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1:
        array = array * 255
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _metric_text(sample: Mapping[str, object]) -> str:
    metrics = sample.get("metrics", {})
    model = metrics.get("model", {}) if isinstance(metrics, Mapping) else {}
    if not isinstance(model, Mapping):
        return ""
    values = []
    for name, label in (("ade", "ADE"), ("fde", "FDE"), ("heading_error", "heading")):
        value = model.get(name)
        if isinstance(value, (float, int)):
            values.append(f"{label}={value:.3f}")
    return "  ".join(values)


def _plot_point(
    point: np.ndarray,
    bounds: tuple[float, float, float, float],
    box: tuple[int, int, int, int],
) -> tuple[int, int]:
    xmin, xmax, ymin, ymax = bounds
    left, top, right, bottom = box
    x, y = point
    px = left + (x - xmin) / max(xmax - xmin, 1e-6) * (right - left)
    py = bottom - (y - ymin) / max(ymax - ymin, 1e-6) * (bottom - top)
    return round(px), round(py)


def _bounds(sample: Mapping[str, object]) -> tuple[float, float, float, float]:
    values = []
    for key in ("target", "prediction", "constant_velocity", "route_only", "obstacles"):
        if sample.get(key) is not None:
            points = _xy(sample[key])
            if len(points):
                values.append(points)
    if not values:
        values.append(np.zeros((1, 2), dtype=np.float32))
    points = np.concatenate(values, axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = max(float(np.max(maximum - minimum)), 1e-3)
    padding = span * 0.15
    return (
        float(minimum[0] - padding),
        float(maximum[0] + padding),
        float(minimum[1] - padding),
        float(maximum[1] + padding),
    )


def _draw_trajectory(
    draw: ImageDraw.ImageDraw,
    sample: Mapping[str, object],
    key: str,
    color: tuple[int, int, int],
    bounds: tuple[float, float, float, float],
    box: tuple[int, int, int, int],
) -> None:
    if sample.get(key) is None:
        return
    points = [_plot_point(point, bounds, box) for point in _xy(sample[key])]
    if len(points) > 1:
        draw.line(points, fill=color, width=5, joint="curve")
    for point in (points[0], points[-1]):
        radius = 6
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=color,
        )


def render_trajectory_png(sample: Mapping[str, object], output_path: Path) -> None:
    """Render one input frame and its local-frame trajectories as a PNG."""
    width, height = 1200, 720
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    dataset = str(sample.get("dataset", "unknown"))
    family, subset = dataset_labels(dataset)
    title = (
        f"{sample.get('trajectory_id', 'unknown')} frame={sample.get('frame', '?')}  "
        f"{family}/{subset}"
    )
    draw.text((24, 18), title, fill="black", font=font)
    metric_text = _metric_text(sample)
    if metric_text:
        draw.text((24, 40), metric_text, fill="black", font=font)

    image_box = (24, 90, 544, 650)
    frame = ImageOps.contain(_image(sample["image"]), (image_box[2] - image_box[0], image_box[3] - image_box[1]))
    image_position = (
        image_box[0] + (image_box[2] - image_box[0] - frame.width) // 2,
        image_box[1] + (image_box[3] - image_box[1] - frame.height) // 2,
    )
    canvas.paste(frame, image_position)
    draw.rectangle(image_box, outline=(100, 100, 100), width=2)
    draw.text((image_box[0], image_box[1] - 18), "input frame", fill="black", font=font)

    plot_box = (650, 110, 1150, 650)
    draw.rectangle(plot_box, outline=(100, 100, 100), width=2)
    draw.text((plot_box[0], plot_box[1] - 18), "local-frame trajectory", fill="black", font=font)
    bounds = _bounds(sample)
    xmin, xmax, ymin, ymax = bounds
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = round(plot_box[0] + fraction * (plot_box[2] - plot_box[0]))
        y = round(plot_box[3] - fraction * (plot_box[3] - plot_box[1]))
        draw.line((x, plot_box[1], x, plot_box[3]), fill=(232, 232, 232), width=1)
        draw.line((plot_box[0], y, plot_box[2], y), fill=(232, 232, 232), width=1)
    origin = _plot_point(np.zeros(2, dtype=np.float32), bounds, plot_box)
    draw.ellipse((origin[0] - 7, origin[1] - 7, origin[0] + 7, origin[1] + 7), fill=(0, 0, 0))
    if sample.get("obstacles") is not None:
        for point in _xy(sample["obstacles"]):
            px, py = _plot_point(point, bounds, plot_box)
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(35, 35, 35))
    for key, color in (
        ("target", (35, 150, 65)),
        ("prediction", (210, 45, 45)),
        ("constant_velocity", (45, 90, 190)),
        ("route_only", (225, 135, 20)),
    ):
        _draw_trajectory(draw, sample, key, color, bounds, plot_box)

    legend = (
        ("target", (35, 150, 65)),
        ("prediction", (210, 45, 45)),
        ("constant velocity", (45, 90, 190)),
        ("route only", (225, 135, 20)),
        ("obstacle", (35, 35, 35)),
    )
    legend_x, legend_y = plot_box[0] + 12, plot_box[1] + 12
    for label, color in legend:
        draw.line((legend_x, legend_y + 5, legend_x + 24, legend_y + 5), fill=color, width=5)
        draw.text((legend_x + 32, legend_y), label, fill="black", font=font)
        legend_y += 18
    draw.text((plot_box[0], plot_box[3] + 8), f"x: {xmin:.1f}..{xmax:.1f} m  y: {ymin:.1f}..{ymax:.1f} m", fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")


def _metadata(sample: Mapping[str, object], file_name: str) -> dict[str, object]:
    dataset = str(sample.get("dataset", "unknown"))
    family, subset = dataset_labels(dataset)
    return {
        "file": file_name,
        "dataset": dataset,
        "dataset_family": str(sample.get("dataset_family", family)),
        "subset": str(sample.get("subset", subset)),
        "robot": str(sample.get("robot", "unknown")),
        "trajectory_id": str(sample.get("trajectory_id", "unknown")),
        "frame": int(sample.get("frame", -1)),
        "metrics": sample.get("metrics", {}),
        **({"selected_collision": sample["selected_collision"]} if "selected_collision" in sample else {}),
    }


def write_visualization_index(
    output_dir: Path, samples: Sequence[Mapping[str, object]], image_files: Sequence[str]
) -> list[dict[str, object]]:
    """Write JSONL metadata and a browsable static HTML index."""
    if len(samples) != len(image_files):
        raise ValueError("samples and image_files must have the same length")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_metadata(sample, file_name) for sample, file_name in zip(samples, image_files)]
    (output_dir / "samples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    cards = []
    for row in rows:
        metrics = row.get("metrics", {})
        model = metrics.get("model", {}) if isinstance(metrics, Mapping) else {}
        ade = model.get("ade", "-") if isinstance(model, Mapping) else "-"
        fde = model.get("fde", "-") if isinstance(model, Mapping) else "-"
        cards.append(
            "<article>"
            f"<h2>{escape(row['dataset_family'] + '/' + row['subset'])} "
            f"{escape(row['trajectory_id'])} frame={row['frame']}</h2>"
            f"<p>ADE={escape(str(ade))} &nbsp; FDE={escape(str(fde))}</p>"
            f"<img src=\"{escape(str(row['file']))}\" loading=\"lazy\">"
            "</article>"
        )
    html = (
        "<!doctype html><meta charset='utf-8'><title>RegNav trajectory visualization</title>"
        "<style>body{font-family:sans-serif;background:#eee;margin:1rem}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(600px,1fr));gap:1rem}"
        "article{background:#fff;padding:.75rem;box-shadow:0 1px 4px #aaa}"
        "img{max-width:100%;height:auto}h2{font-size:1rem;margin:.2rem 0}p{margin:.2rem 0}</style>"
        f"<main>{''.join(cards)}</main>"
    )
    (output_dir / "index.html").write_text(html)
    return rows
