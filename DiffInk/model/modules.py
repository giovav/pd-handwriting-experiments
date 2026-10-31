"""model/modules.py: strati del DiT (embedding, adaLN, attention, feed-forward, blocco)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb


class SinusPositionEmbedding(nn.Module):
    """Embedding sinusoidale del timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvPositionEmbedding(nn.Module):
    """Embedding posizionale convoluzionale; azzera il padding tra le due convoluzioni."""

    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None):  # noqa: F722
        if mask is not None:
            mask = mask[:, None, :]

        x = x.permute(0, 2, 1)
        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
            
        for layer in self.conv1d:
            x = layer(x)
            if mask is not None and isinstance(layer, nn.Mish):
                x = x.masked_fill(~mask, 0.0)
                
        return x.permute(0, 2, 1)


class AdaLayerNormZero(nn.Module):
    """adaLN-Zero: dall'embedding di condizione ricava shift, scale e gate per attention e MLP."""

    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    """adaLN finale: solo shift e scale, prima della proiezione d'uscita."""

    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class FeedForward(nn.Module):
    """Feed-forward con GELU."""

    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    """Attention multi-head; il calcolo è in AttnProcessor."""

    def __init__(
        self,
        processor: AttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        context_pre_only=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Serve PyTorch 2.0 o successivo (scaled_dot_product_attention).")

        self.processor = processor
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if self.context_dim is not None:
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if self.context_pre_only is not None:
                self.to_q_c = nn.Linear(context_dim, self.inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, dim)

    def forward(
        self,
        x: float["b n d"],  # noqa: F722
        c: float["b n d"] = None,  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,
        c_rope=None,
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        else:
            return self.processor(self, x, mask=mask, rope=rope)


class AttnProcessor:
    """Self-attention con F.scaled_dot_product_attention e maschera di padding broadcastabile."""

    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if mask is not None and not bool(mask.all()):
            attn_mask = mask.unsqueeze(1).unsqueeze(1)
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        x = attn.to_out[0](x)
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


class DiTBlock(nn.Module):
    """Blocco del DiT: adaLN, self-attention, feed-forward."""

    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor=AttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


class TimestepEmbedding(nn.Module):
    """
    Embedding del timestep. Con legacy_integer_cast riproduce il cast a intero del codice originale.
    """

    def __init__(self, dim, freq_embed_dim=256, legacy_integer_cast: bool = True):
        super().__init__()
        self.legacy_integer_cast = legacy_integer_cast
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: float["b"]):  # noqa: F821
        if self.legacy_integer_cast:
            if timestep.is_floating_point():
                raise TypeError("Con legacy_integer_cast servono timestep interi da 0 a 999, non float.")
            time_hidden = self.time_embed(timestep)
            time_hidden = time_hidden.to(timestep.dtype)
            return self.time_mlp(time_hidden.float())
        else:
            if not timestep.is_floating_point():
                raise TypeError("Senza legacy_integer_cast servono timestep float in [0, 1], non interi.")
            return self.time_mlp(self.time_embed(timestep))