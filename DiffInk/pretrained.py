"""
pretrained.py: caricamento dei pesi ufficiali di DiffInk (VAE e DiT) nei modelli PaHaW.

Scarta i moduli OCR, stile dello scrittore e text embedding; rinomina i tensori dei
blocchi residui; del layer di ingresso del DiT tiene solo le colonne che leggono il
latente. Si ferma se un layer che dovrebbe venire dal checkpoint resterebbe casuale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from torch import nn


def read_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Legge il checkpoint (weights_only=True) e toglie il prefisso 'module.' di DataParallel."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    state = checkpoint.get("model_state_dict", checkpoint)
    prefix = "module."
    return {(name[len(prefix):] if name.startswith(prefix) else name): tensor
            for name, tensor in state.items()}


@dataclass
class LoadReport:
    """Elenco di tensori caricati, nuovi, tagliati e scartati."""
    loaded: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    discarded: list[str] = field(default_factory=list)
    sliced: list[str] = field(default_factory=list)

    def print(self, title: str, model: nn.Module) -> None:
        sizes = {name: p.numel() for name, p in model.state_dict().items()}
        millions = lambda names: sum(sizes.get(n, 0) for n in names) / 1e6
        print(f"\n=== {title} ===")
        print(f"  caricati dal checkpoint : {len(self.loaded):4d} tensori ({millions(self.loaded):8.2f} M parametri)")
        print(f"  nuovi                   : {len(self.new):4d} tensori ({millions(self.new):8.2f} M parametri)")
        for name in self.new:
            print(f"        NEW      {name}  {tuple(model.state_dict()[name].shape)}")
        for name in self.sliced:
            print(f"        SLICED   {name}")
        prefixes = sorted({name.split(".")[0] for name in self.discarded})
        print(f"  scartati dal checkpoint : {len(self.discarded)} tensori, moduli: {prefixes}")


def _layer_of(name: str) -> str:
    """Nome del layer a cui appartiene un tensore (il nome senza l'ultima parte)."""
    return name.rsplit(".", 1)[0]


def _copy_matching(model: nn.Module, state: dict[str, torch.Tensor], report: LoadReport) -> None:
    """Copia i tensori con nome e shape uguali; un layer si carica tutto o niente."""
    ours = model.state_dict()

    broken_layers = {_layer_of(name) for name, tensor in state.items()
                     if name in ours and ours[name].shape != tensor.shape}

    to_load = {}
    for name, tensor in state.items():
        if name in ours and _layer_of(name) not in broken_layers:
            to_load[name] = tensor
        else:
            report.discarded.append(name)

    model.load_state_dict(to_load, strict=False)

    report.loaded = sorted(to_load)
    report.new = sorted(name for name in ours if name not in to_load)


def _check_new_layers(report: LoadReport, allowed_prefixes: tuple[str, ...], what: str) -> None:
    """Si ferma se mancano tensori diversi da quelli che devono essere nuovi."""
    unexpected = [n for n in report.new if not n.startswith(allowed_prefixes)]
    if unexpected:
        raise RuntimeError(
            f"{what}: questi layer dovrebbero venire dal checkpoint ma manca il tensore o la shape è diversa "
            f"(resterebbero casuali):\n    " + "\n    ".join(unexpected) +
            "\nControllare i valori [PRETRAINED] in configs/pahaw_config.yaml o il checkpoint con inspect_checkpoint.py."
        )


_RESIDUAL_RENAME = {"0": "norm1", "2": "conv1", "3": "norm2", "5": "conv2"}
VAE_NEW_LAYERS = ("encoder.conv_1.", "transformer_decoder.fc.")


def _rename_vae_key(name: str) -> str:
    """
    Nomi dei blocchi residui originali (block.0, block.2, ...) -> nomi attuali (norm1, conv1, ...).
    """
    return re.sub(r"\.block\.(\d)\.", lambda m: "." + _RESIDUAL_RENAME.get(m.group(1), "block." + m.group(1)) + ".", name)


def load_pretrained_vae(vae: nn.Module, path: str) -> LoadReport:
    """
    Carica i pesi ufficiali nel VAE. Restano nuovi solo encoder.conv_1 e transformer_decoder.fc.
    """
    state = read_state_dict(path)
    report = LoadReport()

    deleted = ("ocr_model.", "style_classifier.", "ctc.")
    report.discarded += [n for n in state if n.startswith(deleted)]
    state = {n: t for n, t in state.items() if not n.startswith(deleted)}
    
    state = {_rename_vae_key(n): t for n, t in state.items()}

    _copy_matching(vae, state, report)
    _check_new_layers(report, VAE_NEW_LAYERS, "VAE")
    report.print("VAE: pesi ufficiali caricati", vae)
    return report


DIT_NEW_LAYERS = ("class_embed.", "task_embed.", "rotary_embed.")


def load_pretrained_dit(dit: nn.Module, path: str) -> LoadReport:
    """
    Carica i pesi ufficiali nel DiT. Restano nuovi solo class_embed e task_embed.

    Scarta text_embed e taglia input_embed.proj.weight alle prime latent_dim colonne,
    quelle che moltiplicano il latente.
    """
    state = read_state_dict(path)
    report = LoadReport()

    report.discarded += [n for n in state if n.startswith("text_embed.")]
    state = {n: t for n, t in state.items() if not n.startswith("text_embed.")}

    key = "input_embed.proj.weight"
    latent_dim = dit.input_embed.proj.in_features
    if key in state and state[key].shape[1] > latent_dim:
        state[key] = state[key][:, :latent_dim].clone()
        report.sliced.append(f"{key}: tenute le prime {latent_dim} colonne (senza la parte del testo)")

    _copy_matching(dit, state, report)
    _check_new_layers(report, DIT_NEW_LAYERS, "DiT")
    report.print("DiT: pesi ufficiali caricati", dit)
    return report