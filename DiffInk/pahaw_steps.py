"""
pahaw_steps.py: loss dei due stadi di training e generazione end-to-end.

Stadio 1: VAE da solo. Stadio 2: DiT sui latenti del VAE congelato.
"""

import math

import torch
import torch.nn.functional as F

from model.output_head import decode_output, reconstruction_loss


def vae_training_loss(vae, batch: dict, config):
    """
    Loss del VAE (stadio 1): somma pesata dei tre termini di ricostruzione e della KL.

    La classe non viene usata. Restituisce la loss totale e un dizionario con i singoli termini.
    """
    x, mask = batch["x"], batch["mask"]

    output, mu, logvar, latent_mask = vae(x, mask)

    recon = reconstruction_loss(output, x, mask, config.num_mixture)
    kl = vae.kl_loss(mu, logvar, latent_mask)

    total = (config.gmm_weight * recon["gmm"]
             + config.aux_weight * recon["aux"]
             + config.pen_weight * recon["pen"]
             + config.kl_weight * kl)

    parts = {name: value.item() for name, value in recon.items()}
    parts["kl"] = kl.item()
    return total, parts


def dit_training_loss(dit, vae, diffusion, batch: dict, p_uncond: float = 0.15) -> torch.Tensor:
    """
    Loss del DiT (stadio 2): errore quadratico sul latente pulito, sui soli frame reali.

    Il VAE è congelato. Con probabilità p_uncond la classe è sostituita dalla classe
    nulla, così lo stesso modello impara anche la previsione senza classe che serve alla
    classifier-free guidance. Il task non viene mai nascosto.
    """
    x, mask, y, task = batch["x"], batch["mask"], batch["y"], batch["task"]
    batch_size = x.shape[0]

    with torch.no_grad():
        z0, _, _, latent_mask = vae.encode(x, mask, sample=True)
    keep = latent_mask.unsqueeze(-1).float()

    t = diffusion.sample_timesteps(batch_size)
    z_t, _ = diffusion.add_noise(z0, t)
    z_t = z_t * keep

    hide_class = torch.rand(batch_size, device=y.device) < p_uncond
    y_in = torch.where(hide_class, torch.full_like(y, dit.null_class), y)

    z0_pred = dit(z_t, t, y_in, task, latent_mask)

    squared_error = F.mse_loss(z0_pred, z0, reduction="none")
    return (squared_error * keep).sum() / (keep.sum() * z0.shape[-1]).clamp(min=1.0)


def lengths_to_mask(lengths: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    """Maschera [B, T] dalle lunghezze, con T arrotondato a un multiplo di 8."""
    T = math.ceil(int(lengths.max()) / multiple) * multiple
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]


@torch.no_grad()
def generate(dit, vae, diffusion, y: torch.Tensor, task: torch.Tensor, lengths: torch.Tensor,
             num_mixture: int, cfg_scale: float = 1.0, sampling_timesteps: int = 50):
    """
    Genera registrazioni delle lunghezze date: DDIM sui latenti, decoder del VAE, decode_output.

    Restituisce le feature normalizzate [B, T, 6] e la maschera [B, T]. generate_synthetic.py
    usa invece la decodifica dell'originale (componente estratta a caso).
    """
    dit.eval()
    vae.eval()

    mask = lengths_to_mask(lengths)
    latent_mask = mask.reshape(mask.shape[0], -1, 8).any(dim=2)

    z = diffusion.ddim_sample(dit, latent_mask, y, task,
                              sampling_timesteps=sampling_timesteps, cfg_scale=cfg_scale)
    output = vae.decode(z, mask)
    features = decode_output(output, mask, num_mixture)
    return features, mask