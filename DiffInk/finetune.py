"""
finetune.py: quali pesi di DiffInk si addestrano nel fine-tuning (freezing e adattatori LoRA).

Il DiT ha 238 milioni di pesi e un fold ha 368 registrazioni di training: addestrarlo
tutto rischierebbe di fargli memorizzare i pazienti. Strategie per il DiT:
    "minimal": solo class_embed, task_embed e LoRA sui layer adaLN;
    "lora"   : come minimal, più LoRA su query e value dell'attention e le proiezioni
               di ingresso e uscita sbloccate (default);
    "full"   : tutto addestrabile.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """
    nn.Linear congelato più un adattatore a basso rango: base(x) + scale * up(down(x)).

    up parte da zero, quindi all'inizio l'uscita è quella del layer originale.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

        self.scale = (alpha if alpha is not None else rank) / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.up(self.down(x))


def _wrap(module: nn.Module, attribute: str, rank: int) -> None:
    """Sostituisce module.<name> con un LoRALinear."""
    layer = getattr(module, attribute)
    if isinstance(layer, LoRALinear):
        return
    setattr(module, attribute, LoRALinear(layer, rank))


def configure_dit_finetuning(dit: nn.Module, strategy: str = "lora", rank: int = 8) -> None:
    """
    Congela il DiT e aggiunge gli adattatori secondo la strategia.

    Va chiamata dopo aver caricato i pesi ufficiali.
    """
    if strategy not in ("minimal", "lora", "full"):
        raise ValueError(f"Strategia di fine-tuning sconosciuta: '{strategy}'")

    if strategy == "full":
        for p in dit.parameters():
            p.requires_grad = True
        return

    for p in dit.parameters():
        p.requires_grad = False

    dit.class_embed.weight.requires_grad = True
    dit.task_embed.weight.requires_grad = True

    for block in dit.transformer_blocks:
        _wrap(block.attn_norm, "linear", rank)
    _wrap(dit.norm_out, "linear", rank)

    if strategy == "lora":
        for block in dit.transformer_blocks:
            _wrap(block.attn, "to_q", rank)
            _wrap(block.attn, "to_v", rank)

        for p in dit.input_embed.proj.parameters():
            p.requires_grad = True
        for p in dit.proj_out.parameters():
            p.requires_grad = True


def configure_vae_finetuning(vae: nn.Module, new_names: list[str], phase: str) -> None:
    """
    Pesi addestrabili del VAE nelle due fasi.

    "warmup": solo i due layer nuovi (encoder.conv_1 e transformer_decoder.fc);
    "full"  : tutto il VAE.
    """
    if phase not in ("warmup", "full"):
        raise ValueError(f"Fase di fine-tuning sconosciuta: '{phase}'")
    new = set(new_names)
    for name, p in vae.named_parameters():
        p.requires_grad = (phase == "full") or (name in new)


def param_groups(model: nn.Module, new_names: list[str], lr_new: float,
                 lr_pretrained: float, weight_decay: float) -> list[dict]:
    """
    Due gruppi per AdamW: parametri nuovi (embedding, adattatori, layer nuovi) a lr_new,
    pesi pretrained sbloccati a lr_pretrained. I parametri congelati sono esclusi.
    """
    new = set(new_names)
    is_new = lambda name: name in new or ".down." in name or ".up." in name

    fresh = [p for n, p in model.named_parameters() if p.requires_grad and is_new(n)]
    pretrained = [p for n, p in model.named_parameters() if p.requires_grad and not is_new(n)]

    groups = []
    if fresh:
        groups.append({"params": fresh, "lr": lr_new, "weight_decay": weight_decay})
    if pretrained:
        groups.append({"params": pretrained, "lr": lr_pretrained, "weight_decay": weight_decay})
    return groups


def print_trainable(model: nn.Module, title: str) -> tuple[int, int]:
    """Stampa quanti parametri sono addestrabili."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{title}: {trainable / 1e6:.2f} M parametri addestrabili su {total / 1e6:.2f} M "
          f"({100 * trainable / total:.2f}%)")
    return trainable, total


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Solo i pesi addestrabili: il modello fine-tuned = pesi ufficiali + questo file."""
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in trainable}