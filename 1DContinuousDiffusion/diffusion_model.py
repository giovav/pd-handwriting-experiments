"""
diffusion_model.py: la rete di Luhman e Luhman (2020) adattata a PaHaW.

È la U-Net 1-D originale (strati di luhman_layers.py) senza l'encoder di testo e stile
e senza la cross-attention. Classe e task sono sommati al vettore del livello di rumore
che leggono tutti i layer FiLM. dx, dy usano input_dense e output_dense originali;
pressure, azimuth, altitude entrano ed escono da due layer nuovi (input_new, output_new),
inizializzati a zero e con un learning rate a parte.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from luhman_layers import (ConvSubLayer, Conv1dLast, DecoderLayer, FeedForward, apply_mask, avg_pool,
                           pool_mask, upsample)

N_CONTINUOUS = 5
N_PRETRAINED_INPUTS = 2


class PaHaWDiffusionWriter(nn.Module):
    """U-Net di denoising condizionata. T deve essere un multiplo di 8 (tre pooling)."""

    def __init__(self, config):
        super().__init__()
        c1, c2, c3 = config.c1, config.c2, config.c3
        cond = c1 // 4
        n_new = N_CONTINUOUS - N_PRETRAINED_INPUTS
        
        self.input_dense = nn.Linear(N_PRETRAINED_INPUTS, c1)
        self.input_new = nn.Linear(n_new, c1, bias=False)
        self.sigma_ffn = FeedForward(1, cond, 2048)
        self.class_embed = nn.Embedding(config.num_classes, cond)
        self.task_embed = nn.Embedding(config.num_tasks, cond)
        
        nn.init.zeros_(self.class_embed.weight)
        nn.init.zeros_(self.task_embed.weight)

        self.enc1 = ConvSubLayer(c1, c1)
        self.enc2 = ConvSubLayer(c1, c2)
        self.enc3 = DecoderLayer(c2, 3, pos_factor=4, with_text=False)
        self.enc4 = ConvSubLayer(c2, c3)
        self.enc5 = DecoderLayer(c3, 4, pos_factor=2, with_text=False)
        
        self.skip_conv1 = Conv1dLast(c1, c2)
        self.skip_conv2 = Conv1dLast(c2, c3)
        self.skip_conv3 = Conv1dLast(c3, 2 * c2)
        
        self.att_dense = nn.Linear(c3, 2 * c2)
        self.att_layers = nn.ModuleList([DecoderLayer(2 * c2, 6, with_text=False)
                                         for _ in range(config.num_att_layers)])
                                         
        self.dec3 = ConvSubLayer(2 * c2, c3)
        self.dec2 = ConvSubLayer(c3, c2)
        self.dec1 = ConvSubLayer(c2, c1)
        
        self.output_dense = nn.Linear(c1, N_PRETRAINED_INPUTS)
        self.output_new = nn.Linear(c1, n_new)
        self.pen_dense = nn.Linear(c1, 1)
        
        for layer in (self.input_new, self.output_new):
            for p in layer.parameters():
                nn.init.zeros_(p)

    def condition(self, sigma: torch.Tensor, y: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        """
        Vettore di condizione [B, 1, 32]: sigma_ffn(sqrt(alpha_bar)) + class_embed[y] + task_embed[task].
        """
        noise_level = self.sigma_ffn(sigma.reshape(-1, 1, 1))
        return noise_level + (self.class_embed(y) + self.task_embed(task)).unsqueeze(1)

    def forward(self, x_noisy, sigma, y, task, mask=None):
        """
        x_noisy: [B, T, 5] canali continui con rumore [dx, dy, pressure, azimuth, altitude];
        sigma: [B] sqrt(alpha_bar); y, task: [B]; mask: [B, T].

        Restituisce il rumore previsto [B, T, 5] e il logit "penna sulla superficie" [B, T].
        """
        T = x_noisy.shape[1]
        assert T % 8 == 0, "T deve essere un multiplo di 8"
        
        cond = self.condition(sigma, y, task)
        m1 = mask
        m2 = pool_mask(m1)
        m4 = pool_mask(m2)
        m8 = pool_mask(m4)

        k = N_PRETRAINED_INPUTS
        x = apply_mask(self.input_dense(x_noisy[..., :k]) + self.input_new(x_noisy[..., k:]), m1)
        
        h1 = self.enc1(x, cond, m1)
        h2 = self.enc3(self.enc2(avg_pool(h1), cond, m2), None, cond, mask=m2)
        h3 = self.enc5(self.enc4(avg_pool(h2), cond, m4), None, cond, mask=m4)
        
        x = apply_mask(self.att_dense(avg_pool(h3)), m8)
        for layer in self.att_layers:
            x = layer(x, None, cond, mask=m8)
            
        x = self.dec3(upsample(x) + self.skip_conv3(h3), cond, m4)
        x = self.dec2(upsample(x) + self.skip_conv2(h2), cond, m2)
        x = self.dec1(upsample(x) + self.skip_conv1(h1), cond, m1)
        
        eps = torch.cat([self.output_dense(x), self.output_new(x)], dim=-1)
        return apply_mask(eps, m1), self.pen_dense(x)[..., 0]