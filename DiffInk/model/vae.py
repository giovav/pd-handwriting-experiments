"""
model/vae.py: il VAE di DiffInk adattato a PaHaW.

encode: [B, T, 6] -> latenti [B, T/8, latent_dim]
decode: latenti -> parametri della distribuzione di uscita [B, T, D_out]
"""

import torch
import torch.nn as nn

from utils.mask import downsample_mask, mask_pyramid

from .blocks import Decoder, Encoder, TransformerDecoder
from .output_head import output_dim

COMPRESSION = 8
LOGVAR_MIN, LOGVAR_MAX = -30.0, 20.0


class VAE(nn.Module):
    """VAE senza condizione di classe: la classe entra solo nel DiT."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.num_mixture = config.num_mixture

        self.encoder = Encoder(in_channels=config.in_channels, hidden_dims=config.hidden_dims)

        self.conv_mu = nn.Conv1d(config.hidden_dims[-1], config.latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv1d(config.hidden_dims[-1], config.latent_dim, kernel_size=1)

        if config.latent_dim == config.decoder_dims[0]:
            self.conv_in = nn.Identity()
        else:
            self.conv_in = nn.Conv1d(config.latent_dim, config.decoder_dims[0], kernel_size=1)
            
        self.decoder = Decoder(hidden_dims=config.decoder_dims, out_channels=config.decoder_dims[-1])
        self.transformer_decoder = TransformerDecoder(
            input_dim=config.decoder_dims[-1],
            hidden_dim=config.trans_hidden_dim,
            output_dim=output_dim(config.num_mixture),
            num_layers=config.trans_num_layers,
            num_heads=config.trans_num_heads,
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor, sample: bool = True):
        """
        x: [B, T, C] con T multiplo di 8; mask: [B, T].

        Restituisce z (campionato se sample=True, altrimenti mu), mu, logvar [B, T/8, D]
        e la maschera latente [B, T/8].
        """
        if x.shape[1] % COMPRESSION != 0:
            raise ValueError(
                f"T = {x.shape[1]} non è un multiplo di {COMPRESSION}: il batch va creato con collate_fn."
            )

        masks = mask_pyramid(mask)

        h = x.transpose(1, 2)

        h = self.encoder(h, masks)
        mu = self.conv_mu(h) * masks[8]
        logvar = self.conv_logvar(h)
        
        logvar = logvar + (torch.clamp(logvar, LOGVAR_MIN, LOGVAR_MAX) - logvar).detach()
        logvar = logvar * masks[8]

        if sample:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        z = z * masks[8]

        z, mu, logvar = (t.transpose(1, 2) for t in (z, mu, logvar))
        latent_mask = downsample_mask(mask, COMPRESSION)
        
        return z, mu, logvar, latent_mask

    def decode(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """z: [B, N, D]; mask: [B, T] con T = 8N. Restituisce l'uscita del decoder [B, T, D_out]."""
        assert mask.shape[1] == z.shape[1] * COMPRESSION, "la maschera deve essere a risoluzione piena (T = 8 * N)"
        masks = mask_pyramid(mask)

        h = z.transpose(1, 2)

        h = self.conv_in(h) * masks[8]
        h = self.decoder(h, masks)

        return self.transformer_decoder(h, mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """encode con campionamento, poi decode."""
        z, mu, logvar, latent_mask = self.encode(x, mask, sample=True)
        output = self.decode(z, mask)
        return output, mu, logvar, latent_mask

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, latent_mask: torch.Tensor) -> torch.Tensor:
        """KL rispetto a N(0, 1), mediata sui frame latenti reali."""
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        valid = latent_mask.unsqueeze(-1).float()
        return (kl * valid).sum() / (valid.sum() * mu.shape[-1]).clamp(min=1.0)