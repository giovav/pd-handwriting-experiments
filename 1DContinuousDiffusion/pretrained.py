"""
pretrained.py: dai pesi ufficiali convertiti (luhman_official.pt) alla rete PaHaW.

input_dense e output_dense si copiano così come sono e servono dx, dy; i layer nuovi
(input_new, output_new, class_embed, task_embed) partono da zero; pen_lifts_dense diventa
pen_dense col segno invertito (probabilità di penna sollevata -> logit di penna sulla
superficie); encoder di testo e stile e cross-attention si scartano. Si ferma se un
parametro che non deve essere nuovo resterebbe senza pesi.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

EXPECTED_NEW = ("class_embed.", "task_embed.", "input_new.", "output_new.")
INIT_FORMAT = "pahaw-diffusion-init-v2"


@dataclass
class LoadReport:
    loaded: list[str] = field(default_factory=list)
    flipped: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    discarded: list[str] = field(default_factory=list)

    def print(self, model: nn.Module) -> None:
        sizes = {n: p.numel() for n, p in model.state_dict().items()}
        millions = lambda names: sum(sizes[n] for n in names) / 1e6
        groups = sorted({n.split(".")[0] if n.startswith("text_style") else ".".join(n.split(".")[:-2])
                         for n in self.discarded})
        
        print("\n=== Diffusion1D: pesi ufficiali di Luhman e Luhman (2020) ===")
        print(f"  caricati dal checkpoint: {len(self.loaded):3d} tensori ({millions(self.loaded):5.2f} M parametri), "
              f"di cui {len(self.flipped)} col segno invertito")
        for n in self.flipped:
            print(f"        FLIPPED  {n}: segno invertito (prob. penna sollevata -> logit penna sulla superficie)")
              
        print(f"  nuovi, inizializzati a 0: {len(self.new):3d} tensori ({millions(self.new):5.3f} M parametri)")
        for n in self.new:
            print(f"        NEW      {n}  {tuple(model.state_dict()[n].shape)}")
            
        print(f"  scartati (testo, stile, cross-attention): {len(self.discarded)} tensori in {len(groups)} moduli")


def load_pretrained_diffusion(model: nn.Module, path) -> LoadReport:
    """Carica luhman_official.pt nella rete PaHaW, con le regole descritte sopra."""
    official = torch.load(path, map_location="cpu", weights_only=True)
    ours = model.state_dict()
    report, to_load = LoadReport(), {}

    for name, tensor in official.items():
        target = "pen_dense." + name.split(".")[-1] if name.startswith("pen_lifts_dense.") else name
        if target not in ours:
            report.discarded.append(name)
            continue
            
        if target.startswith("pen_dense."):
            tensor = -tensor
            report.flipped.append(target)
            
        if tensor.shape != ours[target].shape:
            raise ValueError(f"Shape diversa per {target}: checkpoint {tuple(tensor.shape)}, modello {tuple(ours[target].shape)}. "
                             f"c1, c2, c3 e num_att_layers devono essere quelli ufficiali (128, 192, 256, 2).")
                             
        to_load[target] = tensor
        report.loaded.append(target)

    result = model.load_state_dict(to_load, strict=False)
    assert not result.unexpected_keys
    
    report.new = sorted(result.missing_keys)
    unexpected = [n for n in report.new if not n.startswith(EXPECTED_NEW)]
    if unexpected:
        raise ValueError(f"Parametri che resterebbero senza pesi pur non essendo tra quelli nuovi: {unexpected}")
        
    return report


def new_parameter_names(model: nn.Module) -> list[str]:
    """Nomi dei parametri nuovi, per dargli un learning rate diverso."""
    return [n for n, _ in model.named_parameters() if n.startswith(EXPECTED_NEW)]


def load_pahaw_init(model: nn.Module, path) -> dict:
    """
    Carica luhman_pahaw_init.pt (scritto da convert_official_weights.py) controllandone il formato.
    """
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("format") != INIT_FORMAT:
        raise SystemExit(f"{path} non ha il formato atteso "
                         f"(trovato {saved.get('format') if isinstance(saved, dict) else '?'}, atteso {INIT_FORMAT}). "
                         f"Va rigenerato con convert_official_weights.py.")
                         
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return {k: v for k, v in saved.items() if k != "model_state_dict"}