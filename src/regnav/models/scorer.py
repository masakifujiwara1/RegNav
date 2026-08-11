import torch
from torch import nn

from regnav.config import ModelConfig


class TrajectoryScorer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.proposal_projection = nn.Linear(config.num_poses * 3, config.d_model)
        self.context_projection = nn.Linear(
            config.ego_features + config.route_features, config.d_model
        )
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                config.d_model,
                config.num_heads,
                config.d_ffn,
                batch_first=True,
            )
            for _ in range(config.scorer_layers)
        )
        self.component_head = nn.Linear(config.d_model, 4)
        self.register_buffer("weights", torch.tensor(config.scorer_weights))

    def forward(
        self,
        proposals: torch.Tensor,
        scene_registers: torch.Tensor,
        ego: torch.Tensor,
        route_goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.proposal_projection(proposals.flatten(2))
        context = self.context_projection(torch.cat((ego, route_goal), dim=-1))
        tokens = tokens + context[:, None]
        for layer in self.layers:
            tokens = layer(tokens, scene_registers)
        logits = self.component_head(tokens)
        components = torch.cat(
            (logits[..., :1].sigmoid(), logits[..., 1:].tanh()), dim=-1
        )
        return (components * self.weights).sum(dim=-1), components
