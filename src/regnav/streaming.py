import torch

from regnav.contracts import RegNavBatch, RegNavOutput


class CachedSceneInference:
    def __init__(self, model, scene_refresh_interval: int):
        if scene_refresh_interval <= 0:
            raise ValueError("scene_refresh_interval must be positive")
        self.model = model.eval()
        self.scene_refresh_interval = scene_refresh_interval
        self.scene = None
        self.frame_index = 0

    @torch.inference_mode()
    def __call__(self, batch: RegNavBatch) -> RegNavOutput:
        if self.scene is None or self.frame_index % self.scene_refresh_interval == 0:
            self.scene = self.model.encode_scene(batch["image"])
        output = self.model.decode_scene(
            self.scene, batch["ego"], batch["route_goal"]
        )
        self.frame_index += 1
        return output
