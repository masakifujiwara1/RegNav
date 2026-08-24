import torch
from torch import Tensor, nn


_VJEPA2_REPOSITORY = "facebookresearch/vjepa2:45d025f"


class VJEPA2Adapter(nn.Module):
    def __init__(
        self,
        model_name: str,
        image_size: tuple[int, int],
        patch_size: int,
        context_frames: int,
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        if any(size % patch_size for size in image_size):
            raise ValueError("image_size must be divisible by patch_size")
        if context_frames % 2:
            raise ValueError("context_frames must contain complete tubelets")
        if backbone is None:
            backbone, _ = torch.hub.load(
                _VJEPA2_REPOSITORY,
                model_name,
                pretrained=True,
                num_frames=context_frames,
                trust_repo=True,
            )
        self.backbone = backbone

    @property
    def output_dim(self) -> int:
        return self.backbone.embed_dim

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError("V-JEPA input must have shape [B,C,T,H,W]")
        tokens = self.backbone(video)
        if tokens.ndim != 3:
            raise ValueError("V-JEPA backbone must return patch tokens [B,N,D]")
        return tokens
