"""
luhman_layers.py: gli strati di nn.py di Luhman e Luhman (2020, arXiv:2011.06704) riscritti
in PyTorch, con gli stessi nomi degli attributi del codice Keras, così che ogni tensore del
file .h5 ufficiale corrisponda a un solo parametro.

Aggiunte per PaHaW, che non cambiano nulla se non usate: una maschera di padding
(True = step reale) in ogni blocco e l'opzione with_text=False di DecoderLayer, che
salta la cross-attention sul testo. I tensori sono channels-last [B, T, C], come in Keras.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LN_EPS = 1e-6


def layer_norm(x: torch.Tensor) -> torch.Tensor:
    """LayerNormalization di Keras con scale = 1 e shift = 0 (nell'originale sono congelati)."""
    return F.layer_norm(x, x.shape[-1:], eps=LN_EPS)


def apply_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mette a zero le posizioni di padding."""
    return x if mask is None else x * mask.unsqueeze(-1).to(x.dtype)


def pool_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Maschera a metà risoluzione: una posizione è reale se lo è almeno uno dei suoi due step."""
    if mask is None:
        return None
    B, T = mask.shape
    return mask.reshape(B, T // 2, 2).any(dim=-1)


def avg_pool(x: torch.Tensor) -> torch.Tensor:
    """AveragePooling1D(2) su tensori channels-last."""
    B, T, C = x.shape
    return x[:, : T - T % 2].reshape(B, T // 2, 2, C).mean(dim=2)


def upsample(x: torch.Tensor) -> torch.Tensor:
    """UpSampling1D(2): ogni step ripetuto due volte."""
    return x.repeat_interleave(2, dim=1)


def positional_encoding(length: int, channels: int, pos_factor: float, device) -> torch.Tensor:
    """
    Positional encoding sinusoidale calcolato per qualsiasi lunghezza (l'originale ha una
    tabella fissa di 2000 posizioni; sotto quella lunghezza i valori sono identici).
    """
    pos = np.arange(length)[:, None]
    i = np.arange(channels)[None, :]
    angles = pos * (1 / np.power(10000, (2 * (i // 2)) / np.float32(channels))) * pos_factor
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return torch.tensor(angles[None], dtype=torch.float32, device=device)


class FeedForward(nn.Module):
    """ff_network dell'originale."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, hidden)
        self.dense2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.dense2(F.silu(self.dense1(F.silu(x))))


class AffineTransformLayer(nn.Module):
    """
    Layer FiLM: x * gamma(s) + beta(s), con s il vettore del livello di rumore
    (in PaHaW sommato agli embedding di classe e task).
    """

    def __init__(self, cond_dim: int, filters: int):
        super().__init__()
        self.gamma_dense = nn.Linear(cond_dim, filters)
        self.beta_dense = nn.Linear(cond_dim, filters)

    def forward(self, x, sigma):
        return x * self.gamma_dense(sigma) + self.beta_dense(sigma)


class Conv1dLast(nn.Conv1d):
    """nn.Conv1d su tensori channels-last [B, T, C], padding 'same'."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3):
        super().__init__(in_dim, out_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ConvSubLayer(nn.Module):
    """ConvSubLayer dell'originale: convoluzioni con FiLM e connessione residua convoluzionale."""

    def __init__(self, in_dim: int, filters: int, cond_dim: int = 32):
        super().__init__()
        self.affine1 = AffineTransformLayer(cond_dim, filters // 2)
        self.affine2 = AffineTransformLayer(cond_dim, filters)
        self.affine3 = AffineTransformLayer(cond_dim, filters)
        self.conv_skip = Conv1dLast(in_dim, filters)
        self.conv1 = Conv1dLast(in_dim, filters // 2)
        self.conv2 = Conv1dLast(filters // 2, filters)
        self.fc = nn.Linear(filters, filters)

    def forward(self, x, alpha, mask=None):
        x = apply_mask(x, mask)
        x_skip = self.conv_skip(x)
        x = self.conv1(apply_mask(F.silu(x), mask))
        x = self.affine1(x, alpha)
        x = self.conv2(apply_mask(F.silu(x), mask))
        x = self.affine2(x, alpha)
        x = self.fc(F.silu(x))
        x = self.affine3(x, alpha)
        return apply_mask(x + x_skip, mask)


class MultiHeadAttention(nn.Module):
    """Multi-head attention con F.scaled_dot_product_attention e maschera booleana sulle chiavi."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim, self.num_heads = dim, num_heads
        self.wq, self.wk, self.wv = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.dense = nn.Linear(dim, dim)

    def _split(self, x):
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.dim // self.num_heads).transpose(1, 2)

    def forward(self, q, k, v, key_mask: torch.Tensor | None = None):
        B, Tq, _ = q.shape
        q, k, v = self._split(self.wq(q)), self._split(self.wk(k)), self._split(self.wv(v))
        attn_mask = None if key_mask is None else key_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.dense(out.transpose(1, 2).reshape(B, Tq, self.dim))


class DecoderLayer(nn.Module):
    """
    DecoderLayer dell'originale: cross-attention sul testo, self-attention e feed-forward,
    ognuno con FiLM. Con with_text=False la cross-attention viene saltata (x2 = x).
    """

    def __init__(self, dim: int, num_heads: int, pos_factor: float = 1.0, text_dim: int = 384,
                 with_text: bool = True, cond_dim: int = 32):
        super().__init__()
        self.pos_factor, self.with_text = pos_factor, with_text
        if with_text:
            self.text_dense = nn.Linear(text_dim, dim)
            self.mha = MultiHeadAttention(dim, num_heads)
            self.affine0 = AffineTransformLayer(cond_dim, dim)
            self.affine1 = AffineTransformLayer(cond_dim, dim)
        self.mha2 = MultiHeadAttention(dim, num_heads)
        self.ffn = FeedForward(dim, dim, 2 * dim)
        self.affine2 = AffineTransformLayer(cond_dim, dim)
        self.affine3 = AffineTransformLayer(cond_dim, dim)

    def forward(self, x, text, sigma, text_mask=None, mask=None):
        stroke_pe = positional_encoding(x.shape[1], x.shape[2], self.pos_factor, x.device)
        if self.with_text:
            text = self.text_dense(F.silu(text))
            text = self.affine0(layer_norm(text), sigma)
            text_pe = text + positional_encoding(text.shape[1], text.shape[2], 1.0, x.device)
            x2 = self.mha(x + stroke_pe, text_pe, text, key_mask=text_mask)
            x2 = self.affine1(layer_norm(x2), sigma) + x
        else:
            x2 = x
        x2_pe = x2 + stroke_pe
        x3 = self.mha2(x2_pe, x2_pe, x2, key_mask=mask)
        x3 = self.affine2(layer_norm(x2 + x3), sigma)
        x4 = self.ffn(x3) + x3
        return apply_mask(self.affine3(layer_norm(x4), sigma), mask)


class TextStyleEncoder(nn.Module):
    """Encoder di testo e stile dell'originale; serve solo a verificare la conversione dei pesi."""

    def __init__(self, d_model: int = 384, d_ff: int = 768, cond_dim: int = 32):
        super().__init__()
        self.emb = nn.Embedding(73, d_model)
        self.style_ffn = FeedForward(256, d_model, d_ff)
        self.mha = MultiHeadAttention(d_model, 8)
        self.affine1 = AffineTransformLayer(cond_dim, d_model)
        self.affine2 = AffineTransformLayer(cond_dim, d_model)
        self.affine3 = AffineTransformLayer(cond_dim, d_model)
        self.affine4 = AffineTransformLayer(cond_dim, d_model)
        self.text_ffn = FeedForward(d_model, d_model, 2 * d_model)

    def forward(self, text, style, sigma):
        B = style.shape[0]
        style = style.reshape(B, style.shape[1] * 5, style.shape[2] // 5)
        style = self.affine1(layer_norm(self.style_ffn(style)), sigma)
        text = self.affine2(layer_norm(self.emb(text)), sigma)
        text = self.affine3(layer_norm(text + self.mha(text, style, style)), sigma)
        return self.affine4(layer_norm(self.text_ffn(text)), sigma)


class OriginalDiffusionWriter(nn.Module):
    """
    DiffusionWriter originale completo (testo e stile compresi).

    Serve solo a verificare che i pesi convertiti calcolino quello che calcola il codice
    TensorFlow.
    """

    def __init__(self, num_layers: int = 2, c1: int = 128, c2: int = 192, c3: int = 256):
        super().__init__()
        self.input_dense = nn.Linear(2, c1)
        self.sigma_ffn = FeedForward(1, c1 // 4, 2048)
        self.enc1 = ConvSubLayer(c1, c1)
        self.enc2 = ConvSubLayer(c1, c2)
        self.enc3 = DecoderLayer(c2, 3, pos_factor=4)
        self.enc4 = ConvSubLayer(c2, c3)
        self.enc5 = DecoderLayer(c3, 4, pos_factor=2)
        self.skip_conv1 = Conv1dLast(c1, c2)
        self.skip_conv2 = Conv1dLast(c2, c3)
        self.skip_conv3 = Conv1dLast(c3, 2 * c2)
        self.text_style_encoder = TextStyleEncoder(2 * c2, 4 * c2)
        self.att_dense = nn.Linear(c3, 2 * c2)
        self.att_layers = nn.ModuleList([DecoderLayer(2 * c2, 6) for _ in range(num_layers)])
        self.dec3 = ConvSubLayer(2 * c2, c3)
        self.dec2 = ConvSubLayer(c3, c2)
        self.dec1 = ConvSubLayer(c2, c1)
        self.output_dense = nn.Linear(c1, 2)
        self.pen_lifts_dense = nn.Linear(c1, 1)

    def forward(self, strokes, text, sigma, style):
        sigma = self.sigma_ffn(sigma)
        text_mask = text != 0
        text = self.text_style_encoder(text, style, sigma)
        x = self.input_dense(strokes)
        h1 = self.enc1(x, sigma)
        h2 = self.enc3(self.enc2(avg_pool(h1), sigma), text, sigma, text_mask)
        h3 = self.enc5(self.enc4(avg_pool(h2), sigma), text, sigma, text_mask)
        x = self.att_dense(avg_pool(h3))
        for layer in self.att_layers:
            x = layer(x, text, sigma, text_mask)
        x = self.dec3(upsample(x) + self.skip_conv3(h3), sigma)
        x = self.dec2(upsample(x) + self.skip_conv2(h2), sigma)
        x = self.dec1(upsample(x) + self.skip_conv1(h1), sigma)
        return self.output_dense(x), torch.sigmoid(self.pen_lifts_dense(x))