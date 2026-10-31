"""
model/output_head.py: cosa predice il decoder del VAE per ogni step e la loss di ricostruzione.

Per ogni step: 1 logit della penna, una mixture di 20 gaussiane bivariate per (dx, dy)
e una gaussiana diagonale per pressure, azimuth, altitude.
"""

import math

import torch
import torch.nn.functional as F

DX, DY, PRESSURE, AZIMUTH, ALTITUDE, PEN = range(6)
N_AUX = 3

SIGMA_MIN = 0.01
RHO_MAX = 0.99
NLL_CAP = 100.0


def output_dim(num_mixture: int) -> int:
    """Numero di valori predetti per step."""
    return 1 + 6 * num_mixture + 2 * N_AUX


def split_output(output: torch.Tensor, num_mixture: int) -> dict:
    """
    Separa l'uscita del decoder nei parametri: mixture, gaussiane dei canali extra, logit della penna.
    """
    K = num_mixture
    pen_logit = output[..., 0]
    gmm = output[..., 1:1 + 6 * K]
    aux = output[..., 1 + 6 * K:]

    log_pi, mu1, mu2, s1, s2, rho = torch.split(gmm, K, dim=-1)

    return {
        "pen_logit": pen_logit,
        "log_pi": F.log_softmax(log_pi, dim=-1),
        "mu1": mu1,
        "mu2": mu2,
        "sigma1": F.softplus(s1) + SIGMA_MIN,
        "sigma2": F.softplus(s2) + SIGMA_MIN,
        "rho": torch.tanh(rho) * RHO_MAX,
        "aux_mu": aux[..., :N_AUX],
        "aux_sigma": F.softplus(aux[..., N_AUX:]) + SIGMA_MIN,
    }


def _masked_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Media pesata con i pesi della maschera."""
    return (values * weight).sum() / weight.sum().clamp(min=1.0)


def reconstruction_loss(output: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor, num_mixture: int) -> dict:
    """
    Loss di ricostruzione, mediata solo sugli step reali.

    Tre termini: NLL della mixture per (dx, dy) con logsumexp; NLL gaussiana di
    pressure, azimuth, altitude (pressure solo con la penna appoggiata); BCE della penna.
    """
    p = split_output(output, num_mixture)
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
    gmm_nll = torch.clamp(gmm_nll, max=NLL_CAP)
    gmm_loss = _masked_mean(gmm_nll, valid)

    aux_target = target[..., [PRESSURE, AZIMUTH, ALTITUDE]]
    z = (aux_target - p["aux_mu"]) / p["aux_sigma"]
    aux_nll = 0.5 * z ** 2 + torch.log(p["aux_sigma"]) + 0.5 * math.log(2 * math.pi)
    aux_nll = torch.clamp(aux_nll, max=NLL_CAP)

    aux_weight = valid.unsqueeze(-1).repeat(1, 1, N_AUX)
    aux_weight[..., 0] = aux_weight[..., 0] * target[..., PEN]
    aux_loss = _masked_mean(aux_nll, aux_weight)

    pen_bce = F.binary_cross_entropy_with_logits(
        p["pen_logit"], target[..., PEN], reduction="none"
    )
    pen_loss = _masked_mean(pen_bce, valid)

    return {"gmm": gmm_loss, "aux": aux_loss, "pen": pen_loss}


@torch.no_grad()
def decode_output(output: torch.Tensor, mask: torch.Tensor, num_mixture: int,
                  in_air_pressure: float = 0.0) -> torch.Tensor:
    """
    Uscita del decoder -> registrazione [B, T, 6] normalizzata.

    Per (dx, dy) usa il centro della componente più probabile; penna: logit > 0;
    altri canali: le medie. È la decodifica usata per gli errori di ricostruzione.
    """
    p = split_output(output, num_mixture)

    best = p["log_pi"].argmax(dim=-1, keepdim=True)
    dx = torch.gather(p["mu1"], -1, best).squeeze(-1)
    dy = torch.gather(p["mu2"], -1, best).squeeze(-1)

    pen = (p["pen_logit"] > 0).float()
    pressure = torch.where(pen.bool(), p["aux_mu"][..., 0],
                           torch.full_like(pen, in_air_pressure))

    features = torch.stack(
        [dx, dy, pressure, p["aux_mu"][..., 1], p["aux_mu"][..., 2], pen], dim=-1
    )
    return features * mask.unsqueeze(-1)