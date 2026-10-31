"""
model/blocks.py: encoder, decoder e Transformer finale del VAE di DiffInk, con maschera di padding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedGroupNorm(nn.Module):
    """
    GroupNorm(1, C) che calcola media e varianza solo sulle posizioni reali.

    Senza padding coincide con nn.GroupNorm(1, C).
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        num_channels = x.shape[1]

        count = mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0) * num_channels

        mean = (x * mask).sum(dim=(1, 2), keepdim=True) / count
        var = (((x - mean) ** 2) * mask).sum(dim=(1, 2), keepdim=True) / count

        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None] + self.bias[None, :, None]


class Residual(nn.Module):
    """Blocco residuo convoluzionale; azzera il padding dopo ogni strato."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.norm1 = MaskedGroupNorm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = MaskedGroupNorm(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(x, mask)) * mask
        h = self.conv1(h)
        h = F.gelu(self.norm2(h, mask))
        h = self.conv2(h)
        return (x + h) * mask


class ResidualStack(nn.Module):
    """Sequenza di blocchi residui."""

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [Residual(in_channels, hidden_channels) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Encoder(nn.Module):
    """Encoder: tre convoluzioni con stride 2 (T -> T/8), ognuna seguita da blocchi residui."""

    def __init__(self, in_channels: int = 6, hidden_dims=(128, 256, 384)):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, hidden_dims[0], kernel_size=3, stride=2, padding=1)
        self.res_stack_1 = ResidualStack(hidden_dims[0], hidden_dims[0] // 2, num_layers=4)

        self.conv_2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=2, padding=1)
        self.res_stack_2 = ResidualStack(hidden_dims[1], hidden_dims[1] // 2, num_layers=4)

        self.conv_3 = nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=2, padding=1)
        self.res_stack_3 = ResidualStack(hidden_dims[2], hidden_dims[2] // 2, num_layers=4)

    def forward(self, x: torch.Tensor, masks: dict) -> torch.Tensor:
        x = x * masks[1]

        x = self.conv_1(x) * masks[2]
        x = self.res_stack_1(x, masks[2])

        x = self.conv_2(x) * masks[4]
        x = self.res_stack_2(x, masks[4])

        x = self.conv_3(x) * masks[8]
        x = self.res_stack_3(x, masks[8])
        return x


class Decoder(nn.Module):
    """Decoder: tre ConvTranspose1d con stride 2 (T/8 -> T)."""

    def __init__(self, hidden_dims=(384, 256, 128), out_channels: int = 128):
        super().__init__()
        self.res_stack_1 = ResidualStack(hidden_dims[0], hidden_dims[0] // 2, num_layers=4)
        self.deconv_1 = nn.ConvTranspose1d(hidden_dims[0], hidden_dims[1], kernel_size=4, stride=2, padding=1)

        self.res_stack_2 = ResidualStack(hidden_dims[1], hidden_dims[1] // 2, num_layers=4)
        self.deconv_2 = nn.ConvTranspose1d(hidden_dims[1], hidden_dims[2], kernel_size=4, stride=2, padding=1)

        self.res_stack_3 = ResidualStack(hidden_dims[2], hidden_dims[2] // 2, num_layers=4)
        self.deconv_3 = nn.ConvTranspose1d(hidden_dims[2], out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, masks: dict) -> torch.Tensor:
        x = self.res_stack_1(x * masks[8], masks[8])
        x = self.deconv_1(x) * masks[4]

        x = self.res_stack_2(x, masks[4])
        x = self.deconv_2(x) * masks[2]

        x = self.res_stack_3(x, masks[2])
        x = self.deconv_3(x) * masks[1]
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer a risoluzione piena alla fine del decoder, poi il layer d'uscita (output_head).
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 127,
                 num_layers: int = 3, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=~mask)

        return self.fc(x)