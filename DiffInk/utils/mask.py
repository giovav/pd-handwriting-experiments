"""utils/mask.py: maschere di padding. True (o 1.0) = dato reale, False (o 0.0) = padding."""

import torch
import torch.nn.functional as F


def downsample_mask(mask: torch.Tensor, compression_factor: int = 8) -> torch.Tensor:
    """
    Maschera [B, T] -> [B, T // factor]: un frame latente è reale se lo è almeno uno dei suoi step.
    """
    B, T = mask.shape
    assert T % compression_factor == 0, (
        f"T = {T} non è un multiplo di {compression_factor}: il batch va creato con collate_fn."
    )
    blocks = mask.reshape(B, T // compression_factor, compression_factor)
    return blocks.any(dim=2)


def mask_pyramid(mask: torch.Tensor) -> dict[int, torch.Tensor]:
    """
    Maschere float per le risoluzioni T, T/2, T/4, T/8 del VAE.

    Restituisce {1: [B, 1, T], 2: [B, 1, T/2], 4: [B, 1, T/4], 8: [B, 1, T/8]},
    pronte da moltiplicare con feature map [B, C, T].
    """
    full = mask[:, None, :].float()
    pyramid = {1: full}
    for factor in (2, 4, 8):
        pyramid[factor] = F.max_pool1d(full, kernel_size=factor)
    return pyramid