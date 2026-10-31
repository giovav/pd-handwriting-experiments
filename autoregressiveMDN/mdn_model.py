"""
mdn_model.py: la synthesis network di Graves (2013) adattata a PaHaW.

Tre LSTM con peephole; la finestra sul testo (SoftWindow) è sostituita da un vettore
costante di classe e task, che entra nelle stesse righe di W_xh in tutti e tre i layer.
Nomi e shape dei pesi sono quelli del repository di partenza, così il checkpoint si
carica direttamente.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

N_KEPT_INPUTS = 2
N_NEW_INPUTS = 4

State = Tuple[Tensor, Tensor]


def _recurrence(z_in: Tensor, W_h: Tensor, W_ci: Tensor, b_i: Tensor, W_cf: Tensor, b_f: Tensor,
                b_c: Tensor, W_co: Tensor, b_o: Tensor, h: Tensor, c: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Ciclo nel tempo della LSTM con peephole (Graves 2013, eq. 7-11).

    z_in contiene già il contributo dell'ingresso per tutti gli step; nel ciclo si
    aggiunge solo quello di h(t-1).
    """
    outputs: List[Tensor] = []
    for t in range(z_in.size(1)):
        z = z_in[:, t] + h @ W_h
        z_i, z_f, z_c, z_o = z.chunk(4, 1)
        i = torch.sigmoid(z_i + c * W_ci + b_i)
        f = torch.sigmoid(z_f + c * W_cf + b_f)
        c = f * c + i * torch.tanh(z_c + b_c)
        o = torch.sigmoid(z_o + c * W_co + b_o)
        h = o * torch.tanh(c)
        outputs.append(h)
    return torch.stack(outputs, dim=1), h, c


_recurrence_fast = _recurrence


class PeepholeLSTM(nn.Module):
    """LSTM con peephole, con gli stessi nomi e shape dei parametri dell'originale."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        H = hidden_size
        self.W_ci, self.b_i = nn.Parameter(torch.empty(H)), nn.Parameter(torch.empty(H))
        self.W_cf, self.b_f = nn.Parameter(torch.empty(H)), nn.Parameter(torch.empty(H))
        self.b_c = nn.Parameter(torch.empty(H))
        self.W_co, self.b_o = nn.Parameter(torch.empty(H)), nn.Parameter(torch.empty(H))
        self.W_xh = nn.Parameter(torch.empty(input_size + H, 4 * H))
        bound = 1.0 / math.sqrt(H)
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def project(self, inputs: Tensor) -> Tensor:
        """Contributo dell'ingresso ai 4 gate, calcolato per tutti gli step in un solo prodotto."""
        return inputs @ self.W_xh[: self.input_size]

    def initial_state(self, batch_size: int, device) -> State:
        zeros = torch.zeros(batch_size, self.hidden_size, device=device)
        return zeros, zeros.clone()

    def forward(self, z_in: Tensor, state: State) -> Tuple[Tensor, State]:
        h, c = state
        out, h, c = _recurrence_fast(z_in, self.W_xh[self.input_size:], self.W_ci, self.b_i, self.W_cf,
                                     self.b_f, self.b_c, self.W_co, self.b_o, h, c)
        return out, (h, c)


class MixtureOutput(nn.Module):
    """
    Strato d'uscita: mixture per (dx, dy) (pi, mu, sd, ro, dal checkpoint), logit della
    penna e gaussiane per pressure, azimuth, altitude (pen e aux, nuovi).
    """

    def __init__(self, input_size: int, num_mixture: int):
        super().__init__()
        K = num_mixture
        self.pi = nn.Linear(input_size, K)
        self.mu = nn.Linear(input_size, 2 * K)
        self.sd = nn.Linear(input_size, 2 * K)
        self.ro = nn.Linear(input_size, K)
        self.pen = nn.Linear(input_size, 1)
        self.aux = nn.Linear(input_size, 6)
        
        for layer in (self.pen, self.aux):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, h: Tensor) -> dict:
        return {"pi": self.pi(h), "mu": self.mu(h), "sd": self.sd(h), "ro": self.ro(h),
                "pen": self.pen(h), "aux": self.aux(h)}


class PaHaWMDN(nn.Module):
    """
    Synthesis network: 3 LSTM, l'ingresso arriva a tutti i layer e tutti i layer arrivano all'uscita.
    """

    def __init__(self, config):
        super().__init__()
        H, C, K = config.hidden_size, config.cond_dim, config.num_mixture
        assert config.in_channels == N_KEPT_INPUTS + N_NEW_INPUTS, "servono 6 canali [dx, dy, pressure, azimuth, altitude, pen]"
        self.hidden_size, self.cond_dim = H, C

        self.lstm1 = PeepholeLSTM(N_KEPT_INPUTS + C, H)
        self.lstm2 = PeepholeLSTM(N_KEPT_INPUTS + H + C, H)
        self.lstm3 = PeepholeLSTM(N_KEPT_INPUTS + H + C, H)

        self.extra_input = nn.ModuleList([nn.Linear(N_NEW_INPUTS, 4 * H, bias=False) for _ in range(3)])
        self.class_embed = nn.Embedding(config.num_classes, C)
        self.task_embed = nn.Embedding(config.num_tasks, C)
        for p in [*self.extra_input.parameters(), self.class_embed.weight, self.task_embed.weight]:
            nn.init.zeros_(p)

        self.mixture = MixtureOutput(3 * H, K)

    def initial_state(self, batch_size: int, device) -> Tuple[State, State, State]:
        return tuple(lstm.initial_state(batch_size, device) for lstm in (self.lstm1, self.lstm2, self.lstm3))

    def forward(self, x: Tensor, y: Tensor, task: Tensor, state=None):
        """
        x: [B, T, 6] ingressi (la registrazione spostata di uno step); y, task: [B];
        state: stati delle tre LSTM (None per partire da zero).

        Restituisce le uscite grezze dello strato finale e il nuovo stato.
        """
        B, T, _ = x.shape
        if state is None:
            state = self.initial_state(B, x.device)
        cond = (self.class_embed(y) + self.task_embed(task)).unsqueeze(1).expand(B, T, self.cond_dim)
        kept, new = x[..., :N_KEPT_INPUTS], x[..., N_KEPT_INPUTS:]

        z1 = self.lstm1.project(torch.cat([kept, cond], dim=-1)) + self.extra_input[0](new)
        h1, s1 = self.lstm1(z1, state[0])
        z2 = self.lstm2.project(torch.cat([kept, h1, cond], dim=-1)) + self.extra_input[1](new)
        h2, s2 = self.lstm2(z2, state[1])
        z3 = self.lstm3.project(torch.cat([kept, h2, cond], dim=-1)) + self.extra_input[2](new)
        h3, s3 = self.lstm3(z3, state[2])

        raw = self.mixture(torch.cat([h1, h2, h3], dim=-1))
        return raw, (s1, s2, s3)


def teacher_forcing_inputs(x: Tensor) -> Tensor:
    """
    Ingresso per il teacher forcing: la registrazione spostata di uno step a destra,
    con un vettore di zeri al primo posto.
    """
    start = torch.zeros_like(x[:, :1])
    return torch.cat([start, x[:, :-1]], dim=1)


@torch.no_grad()
def generate(model: PaHaWMDN, y: Tensor, task: Tensor, lengths: Tensor, bias: float = 0.0,
             in_air_pressure: float = 0.0, generator: torch.Generator | None = None) -> Tuple[Tensor, Tensor]:
    """
    Generazione senza teacher forcing: si parte dal vettore di zeri, a ogni step si
    campiona dalla distribuzione prevista e il campione diventa l'ingresso dello step
    successivo. Le lunghezze sono date.
    """
    from mdn_head import sample_step

    model.eval()
    B, T = len(y), int(lengths.max())
    device = next(model.parameters()).device
    state = model.initial_state(B, device)
    current = torch.zeros(B, 1, 6, device=device)
    steps = []
    
    for _ in range(T):
        raw, state = model(current, y, task, state)
        step = sample_step(raw, bias, in_air_pressure, generator)
        steps.append(step)
        current = step.unsqueeze(1)
        
    x = torch.stack(steps, dim=1)
    mask = torch.arange(T, device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)
    return x * mask.unsqueeze(-1), mask