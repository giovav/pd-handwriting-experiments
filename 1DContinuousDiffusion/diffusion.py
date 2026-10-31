"""
diffusion.py: rumore, loss e campionamento del modello di Luhman e Luhman (2020).

Come nell'originale: 60 livelli di rumore, alpha_bar estratto in modo continuo tra due
livelli vicini, sampler di inference.py (diffusion_mode='new'); lo stato della penna non
viene diffuso ma predetto, e la sua loss è pesata con alpha_bar. Differenze: i canali
continui sono 5, la loss è mediata solo sugli step reali e la pressione conta solo con
la penna appoggiata.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

N_LEVELS = 60
DX, DY, PRESSURE, AZIMUTH, ALTITUDE, PEN = range(6)


def beta_set(device=None) -> torch.Tensor:
    """beta dei 60 livelli: 0.02 + exp(linspace(log 1e-5, log 0.4)), come nell'originale."""
    return 0.02 + torch.exp(torch.linspace(math.log(1e-5), math.log(0.4), N_LEVELS, device=device))


def alpha_set(device=None) -> torch.Tensor:
    """alpha_bar di ogni livello (prodotto cumulativo di 1 - beta)."""
    return torch.cumprod(1 - beta_set(device), dim=0)


def sample_alpha_bar(batch_size: int, device, generator=None) -> torch.Tensor:
    """Un alpha_bar per elemento del batch, estratto uniformemente tra due livelli vicini."""
    alphas = alpha_set(device)
    i = torch.randint(0, N_LEVELS - 1, (batch_size,), device=device, generator=generator)
    u = torch.rand(batch_size, device=device, generator=generator)
    return alphas[i] + u * (alphas[i + 1] - alphas[i])


def masked_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Media sugli step reali."""
    return (values * weight).sum() / weight.sum().clamp(min=1.0)


def diffusion_loss(model, x: torch.Tensor, mask: torch.Tensor, y, task, generator=None) -> dict:
    """
    Loss di training: si aggiunge rumore ai canali continui a un livello alpha_bar casuale.

    Restituisce tre termini mediati sugli step reali:
        noise_xy : errore quadratico sul rumore previsto per dx, dy;
        noise_aux: lo stesso per pressure, azimuth, altitude (pressure solo con la penna appoggiata);
        pen      : BCE dello stato della penna, moltiplicata per alpha_bar.
    """
    B = x.shape[0]
    a = sample_alpha_bar(B, x.device, generator)
    clean = x[..., :PEN]
    eps = torch.randn(clean.shape, device=x.device, generator=generator)
    noisy = a.sqrt()[:, None, None] * clean + (1 - a).sqrt()[:, None, None] * eps

    eps_pred, pen_logit = model(noisy, a.sqrt(), y, task, mask)
    valid = mask.float()
    squared = (eps - eps_pred) ** 2
    
    aux_weight = torch.ones_like(squared[..., 2:])
    aux_weight[..., 0] = x[..., PEN]

    pen_bce = F.binary_cross_entropy_with_logits(pen_logit, x[..., PEN], reduction="none")
    return {
        "noise_xy": masked_mean(squared[..., :2].sum(-1), valid),
        "noise_aux": masked_mean((squared[..., 2:] * aux_weight).sum(-1), valid),
        "pen": masked_mean(pen_bce * a[:, None], valid),
    }


@torch.no_grad()
def sample(model, y, task, lengths, in_air_pressure: float = 0.0, generator=None):
    """
    Genera registrazioni delle lunghezze date partendo da rumore puro, con il sampler
    originale (60 passi; alpha_bar_next = 1 negli ultimi due). Lo stato della penna è preso
    dall'ultima chiamata della rete; dove la penna è sollevata la pressione è messa al valore
    in aria.
    """
    device = next(model.parameters()).device
    B = len(y)
    T = int(lengths.max())
    T = T + (-T) % 8
    
    mask = torch.arange(T, device=device)[None] < lengths.to(device)[:, None]
    betas, alphas = beta_set(device), alpha_set(device)

    x = torch.randn(B, T, PEN, device=device, generator=generator) * mask[..., None]
    
    for i in range(N_LEVELS - 1, -1, -1):
        a, b = alphas[i], betas[i]
        a_next = alphas[i - 1] if i > 1 else torch.tensor(1.0, device=device)
        eps_pred, pen_logit = model(x, a.sqrt().expand(B), y, task, mask)
        
        x = (x - (1 - a).sqrt() * eps_pred) / (1 - b).sqrt()
        x = x + torch.randn(x.shape, device=device, generator=generator) * (1 - a_next).sqrt()
        x = x * mask[..., None]

    pen = (pen_logit > 0).float()
    x[..., PRESSURE] = torch.where(pen.bool(), x[..., PRESSURE], torch.full_like(pen, in_air_pressure))
    
    out = torch.cat([x, pen.unsqueeze(-1)], dim=-1) * mask[..., None]
    return out, mask