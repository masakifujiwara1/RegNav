import torch
from torch import nn

from regnav.config import ModelConfig
from regnav.contracts import RegNavBatch, RegNavOutput
from regnav.models.lora import inject_attention_lora
from regnav.models.registers import SceneRegisterPool, TrajectoryRegisterDecoder
from regnav.models.scorer import TrajectoryScorer
from regnav.models.vit_adapter import DinoV2Adapter


class RegNav(nn.Module):
    def __init__(self, config: ModelConfig, backbone: nn.Module | None = None):
        super().__init__()
        self.config = config
        if backbone is None:
            adapter = DinoV2Adapter(
                config.backbone, config.image_size, config.patch_size
            )
            backbone = adapter.backbone
        else:
            adapter = DinoV2Adapter(
                config.backbone, config.image_size, config.patch_size, backbone
            )
        inject_attention_lora(backbone, config.lora_rank)
        self.backbone = backbone
        self.image_adapter = adapter
        self.scene_pool = SceneRegisterPool(
            config.num_scene_registers,
            config.backbone_dim,
            config.d_model,
            config.num_heads,
        )
        self.trajectory_embeddings = nn.Parameter(
            torch.randn(config.num_proposals, config.d_model) * 0.02
        )
        self.ego_projection = nn.Linear(config.ego_features, config.d_model)
        self.route_projection = nn.Linear(config.route_features, config.d_model)
        self.decoder = TrajectoryRegisterDecoder(config)
        self.scorer = TrajectoryScorer(config)

    def forward(self, batch: RegNavBatch) -> RegNavOutput:
        scene = self.scene_pool(self.image_adapter(batch["image"]))
        trajectory_tokens = self.trajectory_embeddings[None].expand(
            len(batch["image"]), -1, -1
        )
        trajectory_tokens = (
            trajectory_tokens
            + self.ego_projection(batch["ego"])[:, None]
            + self.route_projection(batch["route_goal"])[:, None]
        )
        refinements = self.decoder(scene, trajectory_tokens)
        proposals = refinements[-1]
        scores, components = self.scorer(
            proposals, scene, batch["ego"], batch["route_goal"]
        )
        trajectory = proposals[
            torch.arange(len(proposals), device=proposals.device), scores.argmax(dim=-1)
        ]
        output: RegNavOutput = {
            "trajectory": trajectory,
            "proposals": proposals,
            "scores": scores,
            "score_components": components,
        }
        if self.training:
            output["refinements"] = refinements
        return output
