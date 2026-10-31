"""
model/dit.py: il Diffusion Transformer di DiffInk, condizionato su classe e task.

Classe e task sono sommati all'embedding del timestep, che ogni blocco legge
attraverso l'adaLN.
"""

from __future__ import annotations

import torch
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from .modules import (
    AdaLayerNormZero_Final,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
)


class InputEmbedding(nn.Module):
    """
    Proiezione dal latente alla larghezza del Transformer, più l'embedding posizionale convoluzionale.
    """

    def __init__(self, latent_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(latent_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.conv_pos_embed(x, mask) + x
        return x


class DiT(nn.Module):
    """DiT con class_embed e task_embed; i nomi dei layer sono quelli del checkpoint ufficiale."""
    
    def __init__(self, config):
        super().__init__()
        dim = config.dim
        self.num_classes = config.num_classes
        self.noise_steps = config.noise_steps

        assert config.time_embedding in ("legacy", "fixed")
        self.legacy_time = config.time_embedding == "legacy"

        self.time_embed = TimestepEmbedding(dim, legacy_integer_cast=self.legacy_time)
        self.class_embed = nn.Embedding(config.num_classes + 1, dim)
        self.task_embed = nn.Embedding(config.num_tasks, dim)

        self.input_embed = InputEmbedding(latent_dim=config.latent_dim, out_dim=dim)
        self.rotary_embed = RotaryEmbedding(config.dim_head)

        self.transformer_blocks = nn.ModuleList([
            DiTBlock(dim=dim, heads=config.heads, dim_head=config.dim_head,
                     ff_mult=config.ff_mult, dropout=config.dropout)
            for _ in range(config.depth)
        ])
        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if config.long_skip_connection else None
        )

        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, config.latent_dim)

        self.initialize_weights()

    @property
    def null_class(self) -> int:
        """Indice della classe nulla, usata per la classifier-free guidance."""
        return self.num_classes

    def initialize_weights(self):
        """Mette a zero adaLN dei blocchi, norma e proiezione finali, embedding di classe e task."""
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

        nn.init.zeros_(self.class_embed.weight)
        nn.init.zeros_(self.task_embed.weight)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        task: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [B, N, D] latente con rumore; t: [B] timestep; y, task: [B]; mask: [B, N].

        Restituisce la previsione del latente pulito, [B, N, D].
        """
        seq_len = x.shape[1]

        time_input = t if self.legacy_time else t.float() / self.noise_steps

        cond = self.time_embed(time_input) + self.class_embed(y) + self.task_embed(task)

        x = self.input_embed(x, mask)
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        residual = x if self.long_skip_connection is not None else None

        for block in self.transformer_blocks:
            x = block(x, cond, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, cond)
        return self.proj_out(x)