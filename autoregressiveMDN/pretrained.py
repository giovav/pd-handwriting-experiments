"""
pretrained.py: caricamento del checkpoint della synthesis network (IAM-OnDB) nell'MDN di PaHaW.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

EOS_ROW = 2
EXPECTED_NEW = ("extra_input.", "class_embed.", "task_embed.", "mixture.pen.", "mixture.aux.")
EXPECTED_DISCARDED = ("window.", "mixture.eos.")


def read_state_dict(path) -> dict[str, torch.Tensor]:
    """Legge il checkpoint con weights_only=True."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state)
    return {name.removeprefix("module."): tensor for name, tensor in state.items()}


@dataclass
class LoadReport:
    loaded: list[str] = field(default_factory=list)
    sliced: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    discarded: list[str] = field(default_factory=list)

    def print(self, model: nn.Module) -> None:
        sizes = {name: p.numel() for name, p in model.state_dict().items()}
        millions = lambda names: sum(sizes[n] for n in names) / 1e6
        print("\n=== MDN: pesi ufficiali (IAM-OnDB) ===")
        print(f"  caricati dal checkpoint : {len(self.loaded):3d} tensori  ({millions(self.loaded):6.2f} M parametri)"
              f", di cui {len(self.sliced)} senza la riga dell'end-of-stroke")
        print(f"  nuovi, inizializzati a 0: {len(self.new):3d} tensori  ({millions(self.new):6.2f} M parametri)")
        for name in self.new:
            print(f"        NEW      {name}  {tuple(model.state_dict()[name].shape)}")
        for name in self.sliced:
            print(f"        SLICED   {name}: tolta la riga dell'end-of-stroke")
        print(f"  scartati dal checkpoint: {len(self.discarded)} tensori (layer non presenti in questo modello): "
              f"{sorted({n.rsplit('.', 1)[0] for n in self.discarded})}")


def load_pretrained_mdn(model: nn.Module, path) -> LoadReport:
    """
    Copia il checkpoint nel modello.

    Vettori peephole, bias e strati della mixture si copiano così come sono; da ogni W_xh
    si toglie la riga 2 (end-of-stroke); window.* e mixture.eos.* si scartano. Si ferma se
    una dimensione della configurazione non coincide con il checkpoint o se un tensore
    diverso da quelli nuovi resterebbe senza pesi.
    """
    state = read_state_dict(path)
    ours = model.state_dict()
    report = LoadReport()
    to_load = {}

    for name, tensor in state.items():
        if name.startswith(EXPECTED_DISCARDED):
            report.discarded.append(name)
            continue
        if name not in ours:
            raise ValueError(f"Tensore sconosciuto nel checkpoint: {name}. È il checkpoint della synthesis network (non quello unconditional)?")
        if name.endswith(".W_xh"):
            tensor = torch.cat([tensor[:EOS_ROW], tensor[EOS_ROW + 1:]], dim=0)
            report.sliced.append(name)
        if tensor.shape != ours[name].shape:
            raise ValueError(
                f"Shape diversa per {name}: checkpoint {tuple(tensor.shape)}, modello {tuple(ours[name].shape)}. "
                f"Le dimensioni [PRETRAINED] della configurazione devono restare quelle del checkpoint.")
        to_load[name] = tensor
        report.loaded.append(name)

    result = model.load_state_dict(to_load, strict=False)
    assert not result.unexpected_keys
    report.new = sorted(result.missing_keys)

    unexpected_new = [n for n in report.new if not n.startswith(EXPECTED_NEW)]
    if unexpected_new:
        raise ValueError(f"Questi layer resterebbero inizializzati a caso perché mancano nel checkpoint:\n {unexpected_new}")
    return report


def new_parameter_names(model: nn.Module) -> list[str]:
    """
    Nomi dei parametri nuovi (non presenti nel checkpoint), per dargli un learning rate diverso.
    """
    return [name for name, _ in model.named_parameters() if name.startswith(EXPECTED_NEW)]