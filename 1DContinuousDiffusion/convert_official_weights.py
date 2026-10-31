"""
convert_official_weights.py: conversione dei pesi ufficiali di Luhman e Luhman (2020) da
Keras (.h5) a PyTorch, e adattamento alla rete PaHaW. Si esegue una volta, in locale.

Stadio 1: weights/model_weights.h5 -> pretrained/luhman_official.pt, usando
weight_map.json; il risultato viene confrontato con le uscite salvate del codice
TensorFlow originale (check 1a) e, con --verify_tf, con il codice originale eseguito al
momento (check 1b).
Stadio 2: luhman_official.pt -> pretrained/luhman_pahaw_init.pt (pretrained.py); si
controlla che la rete PaHaW calcoli esattamente la rete originale senza cross-attention
sul testo (check 2).
Ogni riga stampata finisce anche in pretrained/conversion_report.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

from luhman_layers import OriginalDiffusionWriter

HERE = Path(__file__).resolve().parent
TOLERANCE = 1e-3
REPORT_LINES: list[str] = []


def say(text: str = "") -> None:
    """Stampa e tiene la riga per il report."""
    print(text)
    REPORT_LINES.append(text)


def load_config():
    """Configurazione della rete PaHaW."""
    return types.SimpleNamespace(**yaml.safe_load(open(HERE / "configs" / "pahaw_config.yaml")))


def convert(h5_path: Path) -> dict[str, torch.Tensor]:
    """
    Stadio 1: copia ogni dataset del .h5 nel parametro PyTorch indicato da weight_map.json.

    I kernel Dense si trasponono, quelli Conv1D si permutano da [k, in, out] a [out, in, k].
    """
    import h5py

    table = json.loads((HERE / "weight_map.json").read_text())["parameters"]
    state = {}
    with h5py.File(h5_path, "r") as f:
        for name, entry in table.items():
            value = torch.from_numpy(np.array(f[entry["h5"]], dtype=np.float32))
            if entry["layout"] == "dense":
                value = value.T
            elif entry["layout"] == "conv":
                value = value.permute(2, 1, 0)
            state[name] = value.contiguous()
    return state


@torch.no_grad()
def run_original_pytorch(state: dict, strokes, text, sigma, style):
    """Esegue OriginalDiffusionWriter con i pesi dati sugli ingressi indicati."""
    model = OriginalDiffusionWriter()
    model.load_state_dict(state, strict=True)
    model.eval()
    out, pen = model(torch.as_tensor(strokes), torch.as_tensor(text).long(),
                     torch.as_tensor(sigma), torch.as_tensor(style))
    return out.numpy(), pen.numpy()


def compare_with_tensorflow(state: dict) -> dict[str, float]:
    """
    Check 1a: differenza massima rispetto alle uscite TensorFlow salvate in reference/tf_reference.npz.
    """
    ref = np.load(HERE / "reference" / "tf_reference.npz")
    out, pen = run_original_pytorch(state, ref["strokes"], ref["text"], ref["sigma"], ref["style"])
    return {"noise prediction": float(np.abs(out - ref["out"]).max()),
            "pen-lift probability": float(np.abs(pen - ref["pen"]).max()),
            "(output size, for scale)": float(np.abs(ref["out"]).max())}


def compare_with_live_tensorflow(state: dict, h5_path: Path, seed: int) -> dict[str, float]:
    """
    Check 1b (--verify_tf): confronto con il codice TensorFlow originale eseguito ora, su
    ingressi casuali nuovi. Richiede TensorFlow e tf_keras.
    """
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        import nn as original_nn
    except ImportError as error:
        raise SystemExit(f"--verify_tf richiede TensorFlow e tf_keras installati ({error}).")
        
    tf_model = original_nn.DiffusionWriter(num_layers=2, c1=128, c2=192, c3=256, drop_rate=0.0)
    rng = np.random.default_rng(seed)
    
    B, T, U = 4, 160, 20
    strokes = rng.standard_normal((B, T, 2)).astype("float32")
    text = rng.integers(1, 73, (B, U)).astype("int32")
    text[0, 12:] = 0
    sigma = rng.uniform(0.02, 0.99, (B, 1, 1)).astype("float32")
    style = rng.standard_normal((B, 14, 1280)).astype("float32")
    
    tf_model(strokes, text, sigma, style)
    tf_model.load_weights(str(h5_path))
    tf_out, tf_pen = [t.numpy() for t in tf_model(strokes, text, sigma, style, training=False)[:2]]
    out, pen = run_original_pytorch(state, strokes, text, sigma, style)
    
    return {"noise prediction": float(np.abs(out - tf_out).max()),
            "pen-lift probability": float(np.abs(pen - tf_pen).max()),
            "(output size, for scale)": float(np.abs(tf_out).max())}


@torch.no_grad()
def check_adaptation(official_state: dict, pahaw_state: dict, config) -> dict[str, float]:
    """
    Check 2: la rete PaHaW, con i pesi adattati, calcola quello che calcola la rete originale
    senza cross-attention sul testo: stesso rumore per dx, dy, probabilità della penna
    complementare, zero per i tre canali nuovi.
    """
    from diffusion_model import PaHaWDiffusionWriter

    original = OriginalDiffusionWriter()
    original.load_state_dict(official_state)
    original.eval()
    for block in (original.enc3, original.enc5, *original.att_layers):
        block.with_text = False
        
    model = PaHaWDiffusionWriter(config)
    model.load_state_dict(pahaw_state, strict=True)
    model.eval()

    g = torch.Generator().manual_seed(0)
    B, T = 4, 160
    x = torch.randn(B, T, 5, generator=g)
    sigma = torch.rand(B, generator=g) * 0.95 + 0.02
    y, task = torch.tensor([0, 1, 0, 1]), torch.tensor([0, 3, 5, 7])
    
    ref_eps, ref_lift = original(x[..., :2], torch.ones(B, 4, dtype=torch.long), sigma.reshape(B, 1, 1),
                                 torch.zeros(B, 14, 1280))
    eps, pen_logit = model(x, sigma, y, task)
    
    return {"noise dx, dy": (eps[..., :2] - ref_eps).abs().max().item(),
            "pen probability (ours vs 1 - original)": (torch.sigmoid(pen_logit) - (1 - ref_lift[..., 0])).abs().max().item(),
            "noise of the 3 new channels (must be 0)": eps[..., 2:].abs().max().item()}


def sha256(path: Path) -> str:
    """SHA-256 di un file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(differences: dict, keys: list[str], what: str) -> None:
    """Si ferma se una differenza supera 1e-3."""
    for k, v in differences.items():
        say(f"      {k:45s} {v:.2e}")
    if max(differences[k] for k in keys) > TOLERANCE:
        raise SystemExit(f"FALLITO: {what}. Nessun file scritto.")
    say(f"      -> superato")


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte i pesi ufficiali (Keras) e li adatta alla rete PaHaW.")
    parser.add_argument("--h5", default=str(HERE / "weights" / "model_weights.h5"))
    parser.add_argument("--out_dir", default=str(HERE / "pretrained"))
    parser.add_argument("--verify_tf", action="store_true",
                        help="confronta anche con il codice TensorFlow originale eseguito ora (serve TensorFlow)")
    parser.add_argument("--seed", type=int, default=2026, help="seed degli ingressi casuali del check 1b")
    args = parser.parse_args()

    from pretrained import INIT_FORMAT, load_pretrained_diffusion
    from diffusion_model import PaHaWDiffusionWriter

    h5_path, out_dir = Path(args.h5), Path(args.out_dir)
    if not h5_path.exists():
        raise SystemExit(f"{h5_path} non trovato: serve il file ufficiale weights/model_weights.h5.")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()

    say("=" * 78)
    say("Conversione dei pesi di Luhman e Luhman (2020), HDF5 -> PyTorch")
    say(f"File        : {h5_path}")
    say(f"SHA256      : {sha256(h5_path)}")
    say(f"Versioni    : torch {torch.__version__}, numpy {np.__version__}")
    say("=" * 78)

    say("\nSTAGE 1  Keras -> PyTorch")
    official = convert(h5_path)
    say(f"   {len(official)} tensori, {sum(v.numel() for v in official.values()):,} valori "
        f"(le 10 LayerNorm congelate sono l'identità e non si convertono).")
    say("   Check 1a: confronto con le uscite TensorFlow salvate")
    require(compare_with_tensorflow(official), ["noise prediction", "pen-lift probability"],
            "la rete convertita non riproduce le uscite salvate del codice TensorFlow")
            
    if args.verify_tf:
        say(f"   Check 1b: confronto con il codice TensorFlow eseguito ora (seed {args.seed})")
        require(compare_with_live_tensorflow(official, h5_path, args.seed),
                ["noise prediction", "pen-lift probability"],
                "la rete convertita non riproduce il codice TensorFlow eseguito ora")
                
    official_path = out_dir / "luhman_official.pt"
    torch.save(official, official_path)
    say(f"   Salvato: {official_path}")

    say("\nSTAGE 2  rete ufficiale -> rete PaHaW")
    model = PaHaWDiffusionWriter(config)
    report = load_pretrained_diffusion(model, official_path)
    say(f"   tensori caricati: {len(report.loaded)}; col segno invertito: {report.flipped}")
    say(f"   nuovi (a zero): {report.new}; scartati: {len(report.discarded)} (testo e stile)")
    
    pahaw_state = model.state_dict()
    say(f"   Rete PaHaW: {sum(v.numel() for v in pahaw_state.values()):,} parametri.")
    say("   Check 2: la rete PaHaW calcola la rete originale senza cross-attention sul testo")
    require(check_adaptation(official, pahaw_state, config),
            ["noise dx, dy", "pen probability (ours vs 1 - original)", "noise of the 3 new channels (must be 0)"],
            "la rete PaHaW non coincide con la rete originale senza cross-attention")

    init_path = out_dir / "luhman_pahaw_init.pt"
    torch.save({"format": INIT_FORMAT,
                "model_state_dict": pahaw_state,
                "source_h5_sha256": sha256(h5_path),
                "loaded": report.loaded, "flipped": report.flipped,
                "new": report.new, "discarded": report.discarded}, init_path)
    say(f"   Salvato: {init_path}   <- file di partenza per il training")

    report_path = out_dir / "conversion_report.txt"
    report_path.write_text("\n".join(REPORT_LINES) + "\n")
    print(f"\nConversione verificata. Report: {report_path}")


if __name__ == "__main__":
    main()