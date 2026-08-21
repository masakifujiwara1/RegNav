from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from PIL import Image
import torch

from regnav.config import ModelConfig
from regnav.data.vint_dataset import preprocess_image
from regnav.models.factory import build_model
from regnav.training.engine import load_checkpoint


def _finite_vector(name: str, values: Sequence[float], size: int) -> torch.Tensor:
    tensor = torch.as_tensor(list(values), dtype=torch.float32)
    if tensor.shape != (size,):
        raise ValueError(f"{name} must contain exactly {size} values")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain finite values")
    return tensor


class RegNavPredictor:
    """Load one checkpoint and reuse it for repeated image-to-trajectory calls."""

    def __init__(self, checkpoint: Path, device: str | torch.device | None = None):
        checkpoint = Path(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = ModelConfig(**payload["model_config"])
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = build_model(self.config).to(self.device).eval()
        load_checkpoint(checkpoint, self.model, weights_only=True, map_location=self.device)

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        ego: Sequence[float],
        route_goal: Sequence[float],
    ) -> dict[str, object]:
        batch = {
            "image": preprocess_image(image, self.config.image_size)[None].to(self.device),
            "ego": _finite_vector("ego", ego, self.config.ego_features)[None].to(self.device),
            "route_goal": _finite_vector(
                "route_goal", route_goal, self.config.route_features
            )[None].to(self.device),
        }
        output = self.model(batch)
        selected = int(output["scores"].argmax(dim=-1)[0])
        return {
            "trajectory": output["trajectory"][0].cpu().tolist(),
            "scores": output["scores"][0].cpu().tolist(),
            "selected_proposal": selected,
            "device": str(self.device),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RegNav on one camera image")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--ego", type=float, nargs=4, default=(0.0, 0.0, 0.0, 0.0))
    parser.add_argument("--route-goal", type=float, nargs=6, required=True)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictor = RegNavPredictor(args.checkpoint, args.device)
    with Image.open(args.image) as image:
        result = predictor.predict(image, args.ego, args.route_goal)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
