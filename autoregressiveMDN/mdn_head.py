"""
mdn_head.py: parametri dell'uscita, loss e campionamento di un passo per l'MDN.

Per ogni step: mixture di 20 gaussiane bivariate per (dx, dy) (Graves 2013), una
gaussiana per ciascuno di pressure, azimuth, altitude e un logit per lo stato della
penna. Limiti e cap della loss sono gli stessi del decoder di DiffInk, così le loss dei
due modelli si misurano allo stesso modo.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

DX, DY, PRESSURE, AZIMUTH, ALTITUDE, PEN = range(6)
N_AUX = 3

SIGMA_MIN = 0.01
RHO_MAX = 0.99
NLL_CAP = 100.0
LOG_SIGMA_MAX = 5.0


def positive_width(raw: torch.Tensor) -> torch.Tensor:
    """Larghezza delle gaussiane: exp(raw), con raw al massimo LOG_SIGMA_MAX, più SIGMA_MIN."""
    return torch.exp(torch.clamp(raw, max=LOG_SIGMA_MAX)) + SIGMA_MIN


def mixture_parameters(raw: dict, bias: float = 0.0) -> dict:
    """
    Uscite grezze -> parametri di mixture e gaussiane.

    bias è il sampling bias di Graves (2013), eq. 61-62: pesi della mixture più netti e
    larghezze ridotte. Con bias = 0 si campiona la distribuzione appresa.
    """
    K = raw["pi"].shape[-1]
    mu1, mu2 = raw["mu"][..., :K], raw["mu"][..., K:]
    sd1, sd2 = raw["sd"][..., :K] - bias, raw["sd"][..., K:] - bias
    aux = raw["aux"]
    return {
        "log_pi": F.log_softmax(raw["pi"] * (1.0 + bias), dim=-1),
        "mu1": mu1,
        "mu2": mu2,
        "sigma1": positive_width(sd1),
        "sigma2": positive_width(sd2),
        "rho": torch.tanh(raw["ro"]) * RHO_MAX,
        "pen_logit": raw["pen"][..., 0],
        "aux_mu": aux[..., :N_AUX],
        "aux_sigma": positive_width(aux[..., N_AUX:] - bias),
    }


def _masked_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Media sugli step reali (pesi della maschera)."""
    return (values * weight).sum() / weight.sum().clamp(min=1.0)


def mdn_loss(raw: dict, target: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    Loss dell'MDN, mediata sugli step reali. Restituisce tre termini:
        gmm: NLL della mixture per (dx, dy);
        aux: NLL gaussiana di pressure, azimuth, altitude (pressure solo con la penna appoggiata);
        pen: BCE dello stato della penna.
    Uno step contribuisce al massimo NLL_CAP.
    """
    p = mixture_parameters(raw)
    valid = mask.float()

    z1 = (target[..., DX].unsqueeze(-1) - p["mu1"]) / p["sigma1"]
    z2 = (target[..., DY].unsqueeze(-1) - p["mu2"]) / p["sigma2"]
    one_minus_rho2 = 1.0 - p["rho"] ** 2
    
    log_gauss = (
        -(z1 ** 2 + z2 ** 2 - 2 * p["rho"] * z1 * z2) / (2 * one_minus_rho2)
        - torch.log(2 * math.pi * p["sigma1"] * p["sigma2"])
        - 0.5 * torch.log(one_minus_rho2)
    )
    
    gmm_nll = -torch.logsumexp(p["log_pi"] + log_gauss, dim=-1)
    gmm_loss = _masked_mean(torch.clamp(gmm_nll, max=NLL_CAP), valid)

    aux_target = target[..., [PRESSURE, AZIMUTH, ALTITUDE]]
    z = (aux_target - p["aux_mu"]) / p["aux_sigma"]
    aux_nll = 0.5 * z ** 2 + torch.log(p["aux_sigma"]) + 0.5 * math.log(2 * math.pi)
    
    aux_weight = valid.unsqueeze(-1).repeat(1, 1, N_AUX)
    aux_weight[..., 0] = aux_weight[..., 0] * target[..., PEN]
    aux_loss = _masked_mean(torch.clamp(aux_nll, max=NLL_CAP), aux_weight)

    pen_bce = F.binary_cross_entropy_with_logits(p["pen_logit"], target[..., PEN], reduction="none")
    pen_loss = _masked_mean(pen_bce, valid)

    return {"gmm": gmm_loss, "aux": aux_loss, "pen": pen_loss}


@torch.no_grad()
def sample_step(raw: dict, bias: float = 0.0, in_air_pressure: float = 0.0,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Campiona un passo [B, 6]: componente, coppia (dx, dy) correlata, penna e canali extra.

    Dove la penna risulta sollevata la pressione è messa al valore in aria.
    """
    p = {k: v[:, 0] for k, v in mixture_parameters(raw, bias).items()}
    B = p["log_pi"].shape[0]
    
    component = torch.multinomial(p["log_pi"].exp(), 1, generator=generator)
    pick = lambda name: torch.gather(p[name], 1, component).squeeze(1)
    mu1, mu2, s1, s2, rho = pick("mu1"), pick("mu2"), pick("sigma1"), pick("sigma2"), pick("rho")

    e1 = torch.randn(B, generator=generator, device=mu1.device)
    e2 = torch.randn(B, generator=generator, device=mu1.device)
    dx = mu1 + s1 * e1
    dy = mu2 + s2 * (rho * e1 + torch.sqrt(1 - rho ** 2) * e2)

    pen = torch.bernoulli(torch.sigmoid(p["pen_logit"]), generator=generator)
    aux = p["aux_mu"] + p["aux_sigma"] * torch.randn(p["aux_mu"].shape, generator=generator,
                                                     device=mu1.device)
    
    pressure = torch.where(pen.bool(), aux[:, 0], torch.full_like(pen, in_air_pressure))
    
    return torch.stack([dx, dy, pressure, aux[:, 1], aux[:, 2], pen], dim=-1)