from torch import Tensor, nn


class DinoV2Adapter(nn.Module):
    def __init__(
        self,
        model_name: str,
        image_size: tuple[int, int],
        patch_size: int,
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        if any(size % patch_size for size in image_size):
            raise ValueError("image_size must be divisible by patch_size")
        if backbone is None:
            import timm

            backbone = timm.create_model(
                model_name, pretrained=True, num_classes=0, global_pool=""
            )
        self.backbone = backbone

    @property
    def output_dim(self) -> int:
        return self.backbone.embed_dim

    def forward(self, image: Tensor) -> Tensor:
        tokens = self.backbone.forward_features(image)
        return tokens[:, self.backbone.num_prefix_tokens :]
