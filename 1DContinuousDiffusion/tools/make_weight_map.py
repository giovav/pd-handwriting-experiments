"""
tools/make_weight_map.py: costruisce weight_map.json e reference/tf_reference.npz.

Non serve per il training: si esegue una volta, con TensorFlow e il codice originale
(nn.py di tcl9876/Diffusion-Handwriting-Generation).
    1. Costruisce il modello TensorFlow originale e carica il .h5 ufficiale con
       model.load_weights, come fanno gli autori.
    2. Per ogni parametro PyTorch di luhman_layers.py segue lo stesso percorso di
       attributi nel modello Keras (Keras numera i layer con contatori globali, quindi
       i nomi non bastano).
    3. Cerca il dataset del .h5 con valori identici a quella variabile e si ferma se
       ne trova zero o più di uno.
    4. Salva le uscite del modello originale su ingressi casuali fissi, per poter
       verificare la conversione senza TensorFlow.
"""

import json
import os
import sys
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import h5py
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from luhman_layers import OriginalDiffusionWriter


def keras_object(model, torch_name: str):
    """
    Dal nome di un parametro PyTorch alla variabile Keras corrispondente e al suo layout
    (dense o conv), seguendo gli attributi (ModuleList -> lista Python, FeedForward ->
    Sequential, ...).
    """
    *path, leaf = torch_name.split(".")
    obj, i = model, 0
    while i < len(path):
        part = path[i]
        if part == "att_layers":
            obj, i = obj.att_layers[int(path[i + 1])], i + 2
            continue
        if part in ("dense1", "dense2"):
            obj = obj.layers[1 if part == "dense1" else 2]
        else:
            obj = getattr(obj, part)
        i += 1
        
    if leaf == "bias":
        return obj.bias, "vector"
    if hasattr(obj, "embeddings"):
        return obj.embeddings, "embedding"
        
    kernel = obj.kernel
    return kernel, ("conv" if len(kernel.shape) == 3 else "dense")


def main(repo: Path) -> None:
    """Costruisce la tabella e salva le uscite di riferimento."""
    sys.path.insert(0, str(repo))
    import nn as original_nn
    weights_file = repo / "weights" / "model_weights.h5"

    model = original_nn.DiffusionWriter(num_layers=2, c1=128, c2=192, c3=256, drop_rate=0.0)
    rng = np.random.default_rng(0)
    
    B, T, U = 3, 96, 12
    strokes = rng.standard_normal((B, T, 2)).astype("float32")
    text = rng.integers(1, 73, (B, U)).astype("int32")
    text[1, 9:] = 0
    sigma = np.array([0.05, 0.5, 0.97], "float32").reshape(B, 1, 1)
    style = rng.standard_normal((B, 14, 1280)).astype("float32")
    
    model(strokes, text, sigma, style)
    model.load_weights(str(weights_file))

    datasets = {}
    with h5py.File(weights_file, "r") as f:
        f.visititems(lambda name, obj: datasets.__setitem__(name, np.array(obj)) if isinstance(obj, h5py.Dataset) else None)

    table = {}
    for torch_name, _ in OriginalDiffusionWriter().named_parameters():
        variable, layout = keras_object(model, torch_name)
        value = variable.numpy()
        matches = [n for n, d in datasets.items() if d.shape == value.shape and np.array_equal(d, value)]
        if len(matches) != 1:
            raise SystemExit(f"{torch_name}: {len(matches)} dataset con gli stessi valori (ne deve corrispondere esattamente uno).")
        table[torch_name] = {"h5": matches[0], "layout": layout}
        
    unused = sorted(set(datasets) - {t["h5"] for t in table.values()})
    assert all("layer_normalization" in n for n in unused), unused
    
    for n in unused:
        assert np.all(datasets[n] == (1.0 if n.endswith("gamma:0") else 0.0)), n

    (HERE / "weight_map.json").write_text(json.dumps(
        {"source": "tcl9876/Diffusion-Handwriting-Generation weights/model_weights.h5",
         "unused_identity_layernorm_tensors": unused, "parameters": table}, indent=1))

    out, pen = model(strokes, text, sigma, style, training=False)[:2]
    (HERE / "reference").mkdir(exist_ok=True)
    
    np.savez_compressed(HERE / "reference" / "tf_reference.npz", strokes=strokes, text=text, sigma=sigma,
                        style=style, out=out.numpy(), pen=pen.numpy())
    print(f"{len(table)} parametri PyTorch mappati; {len(unused)} LayerNorm (identità) non usate; uscite di riferimento in tf_reference.npz.")


if __name__ == "__main__":
    main(Path(sys.argv[1]))