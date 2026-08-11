import torch
from torch import nn

from regnav.config import ModelConfig


class SceneRegisterPool(nn.Module):
    def __init__(
        self, num_registers: int, input_dim: int, d_model: int, num_heads: int
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_registers, d_model) * 0.02)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        memory = self.input_projection(patch_tokens)
        queries = self.queries[None].expand(len(patch_tokens), -1, -1)
        return self.attention(queries, memory, memory, need_weights=False)[0]


class TrajectoryRegisterDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                config.d_model,
                config.num_heads,
                config.d_ffn,
                batch_first=True,
            )
            for _ in range(config.decoder_layers)
        )
        self.heads = nn.ModuleList(
            nn.Linear(config.d_model, config.num_poses * 3)
            for _ in range(config.num_refinements)
        )

    def forward(
        self, scene_registers: torch.Tensor, trajectory_registers: torch.Tensor
    ) -> list[torch.Tensor]:
        tokens = trajectory_registers
        refinements = []
        layers_per_head = len(self.layers) // len(self.heads)
        for layer_index, layer in enumerate(self.layers, 1):
            tokens = layer(tokens, scene_registers)
            if layer_index % layers_per_head == 0:
                deltas = self.heads[len(refinements)](tokens).view(
                    len(tokens), self.config.num_proposals, self.config.num_poses, 3
                )
                poses = torch.cumsum(deltas, dim=2)
                poses = torch.cat(
                    (
                        poses[..., :2],
                        torch.atan2(torch.sin(poses[..., 2:]), torch.cos(poses[..., 2:])),
                    ),
                    dim=-1,
                )
                refinements.append(poses)
        return refinements
