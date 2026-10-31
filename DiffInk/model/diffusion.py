"""
model/diffusion.py: processo di diffusione sui latenti (rumore in avanti, campionamento DDIM)
con classifier-free guidance sulla classe clinica.
"""

import math

import torch


class Diffusion:
    """Schedule del rumore, aggiunta del rumore e campionamento DDIM."""

    def __init__(self, noise_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
                 device=None, schedule_type: str = "cosine"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.schedule_type = schedule_type.lower()

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self) -> torch.Tensor:
        """Valori di beta dello schedule scelto (linear o cosine)."""
        if self.schedule_type == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        if self.schedule_type == "cosine":
            return self.cosine_beta_schedule(self.noise_steps)
        raise ValueError(f"schedule_type non supportato: {self.schedule_type}")

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Schedule cosine (Nichol e Dhariwal, 2021)."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def sample_timesteps(self, n: int) -> torch.Tensor:
        """Timestep casuali uniformi, uno per elemento del batch."""
        return torch.randint(0, self.noise_steps, (n,), device=self.device)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Aggiunge rumore al latente pulito.

        x0: [B, N, D], t: [B]. Restituisce x_t e il rumore usato, entrambi [B, N, D].
        """
        assert x0.dim() == 3, "il latente deve avere forma [B, N, D]"
        
        signal_scale = torch.sqrt(self.alpha_hat[t])[:, None, None]
        noise_scale = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None]

        noise = torch.randn_like(x0)
        x_t = signal_scale * x0 + noise_scale * noise
        return x_t, noise

    @torch.no_grad()
    def ddim_sample(self, dit, latent_mask: torch.Tensor, y: torch.Tensor, task: torch.Tensor,
                    sampling_timesteps: int = 50, eta: float = 0.0,
                    cfg_scale: float = 1.0) -> torch.Tensor:
        """
        Genera latenti partendo da rumore puro, con DDIM.

        latent_mask: [B, N], True sui frame reali (i frame di padding restano a zero).
        y, task: [B]. cfg_scale: forza della classifier-free guidance
        (1 = modello condizionato così com'è, > 1 = spinta più forte verso la classe).
        Restituisce i latenti [B, N, D], da passare a VAE.decode.
        """
        dit.eval()
        device = latent_mask.device
        B, N = latent_mask.shape
        keep = latent_mask.unsqueeze(-1).float()

        y_null = torch.full_like(y, dit.null_class)

        x = torch.randn(B, N, dit.proj_out.out_features, device=device) * keep

        times = torch.linspace(-1, self.noise_steps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))

        for time, time_next in zip(times[:-1], times[1:]):
            t = torch.full((B,), time, device=device, dtype=torch.long)

            x0 = dit(x, t, y, task, latent_mask)

            if cfg_scale != 1.0:
                x0_uncond = dit(x, t, y_null, task, latent_mask)
                x0 = x0_uncond + cfg_scale * (x0 - x0_uncond)

            x0 = x0 * keep

            if time_next < 0:
                x = x0
                break

            alpha_hat_t = self.alpha_hat[t][:, None, None]
            t_next = torch.full((B,), time_next, device=device, dtype=torch.long)
            alpha_hat_next = self.alpha_hat[t_next][:, None, None]

            eps = (x - alpha_hat_t.sqrt() * x0) / (1 - alpha_hat_t).clamp_min(1e-8).sqrt()

            sigma = eta * torch.sqrt(
                ((1 - alpha_hat_next) / (1 - alpha_hat_t).clamp_min(1e-8)
                 * (1 - alpha_hat_t / alpha_hat_next)).clamp_min(0.0)
            )
            c = torch.sqrt((1 - alpha_hat_next - sigma ** 2).clamp_min(0.0))
            fresh_noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

            x = (alpha_hat_next.sqrt() * x0 + c * eps + sigma * fresh_noise) * keep

        return x