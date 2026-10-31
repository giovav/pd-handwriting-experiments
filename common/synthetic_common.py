"""
synthetic_common.py: parti comuni ai generate_synthetic.py dei tre modelli: cosa generare,
come controllarlo e come salvarlo.

Piano (uguale per i tre modelli, per ogni fold): per ogni coppia (classe, task) tante
registrazioni quante ne ha il dataset reale (36, quindi 576 in tutto). Ogni lunghezza è
estratta con reinserimento dalle registrazioni di training dello stesso fold, classe e
task, mai da validazione o test. L'estrazione dipende solo da fold e seed: i tre modelli
ricevono le stesse lunghezze, e le differenze tra i loro dataset vengono solo da ciò che
generano.

Pseudo-pazienti: la registrazione k di una classe è salvata con ID "SYN-H-007" o
"SYN-PD-007" (k = 7), con una registrazione per task come un paziente reale. È solo un
modo per riusare il codice scritto per i dati reali: le 8 registrazioni di uno
pseudo-paziente sono generate in modo indipendente e non vanno trattate come la stessa
persona.

Output, in unità originali come i dati reali:
    <out_dir>/<modello>/fold<k><tag>.npz   scritto con packed_data.save_packed
    <out_dir>/<modello>/fold<k><tag>.json  modello, checkpoint, impostazioni, seed, piano, controlli
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import train_common as tc
from train_common import dp
from create_split import load_fold
from packed_data import load_packed, save_packed

CLASS_NAME = {0: "H", 1: "PD"}


# =============================================================================
# Piano di generazione
# =============================================================================

def make_plan(recordings: list[dict], split: dict, fold: int, n_per_cell: int | None, seed: int) -> list[dict]:
    """
    Un elemento per registrazione da generare: {"patient_id", "label", "task", "length"}.

    Con n_per_cell=None, tante registrazioni per (classe, task) quante nel dataset reale.
    """
    train_ids = set(split["train"])
    train_lengths, real_count = defaultdict(list), defaultdict(int)
    for r in recordings:
        cell = (r["label"], r["task"])
        real_count[cell] += 1
        if r["patient_id"] in train_ids:
            train_lengths[cell].append(len(r["features"]))

    rng = np.random.default_rng([seed, fold])          # dipende solo da fold e seed, non dal modello
    plan = []
    for (label, task) in sorted(real_count):
        pool = np.array(sorted(train_lengths[(label, task)]))
        if len(pool) == 0:
            raise SystemExit(f"Fold {fold}: nessuna registrazione di training per classe {label}, task {task + 1}.")
        n = n_per_cell or real_count[(label, task)]
        for k, length in enumerate(rng.choice(pool, size=n, replace=True)):
            plan.append({"patient_id": f"SYN-{CLASS_NAME[label]}-{k:03d}", "label": int(label),
                         "task": int(task), "length": int(length)})
    return plan


def batches(plan: list[dict], max_batch: int, cost, budget: float) -> list[list[int]]:
    """
    Raggruppa gli elementi del piano per lunghezza simile.

    Ogni batch ha al massimo max_batch elementi e cost(dimensione, lunghezza massima) <= budget;
    un elemento che da solo supera il budget forma un batch a sé.
    """
    order = sorted(range(len(plan)), key=lambda i: plan[i]["length"])
    out, current = [], []
    for i in order:
        if current and (len(current) == max_batch or cost(len(current) + 1, plan[i]["length"]) > budget):
            out.append(current)
            current = []
        current.append(i)
    return out + ([current] if current else [])


# =============================================================================
# Dati, normalizzazione, checkpoint
# =============================================================================

def load_fold_data(fold: int, data_file: str, split_file: str):
    recordings = tc.load_recordings(data_file)
    split = load_fold(fold, split_file) if split_file else load_fold(fold)
    return recordings, split


def fold_stats(run_dir: Path, recordings: list[dict], split: dict):
    """
    Costanti di normalizzazione usate nel training (norm_stats.json nella cartella del run).

    Vengono confrontate con quelle ricalcolate dai pazienti di training del fold: se non
    coincidono, il run è stato addestrato su altri dati.
    """
    path = run_dir / "norm_stats.json"
    if not path.exists():
        raise SystemExit(f"{path} non trovato: {run_dir} è la cartella del run di questo fold?")
    saved = dp.NormStats.load(path)
    fresh = dp.fit_norm_stats(recordings, split["train"])
    if not (np.allclose(saved.mean, fresh.mean, rtol=1e-5, atol=1e-6) and np.allclose(saved.std, fresh.std, rtol=1e-5)):
        raise SystemExit(f"{path} non corrisponde al file di dati indicato: il modello è stato addestrato su altri dati. "
                         f"Usare il file di dati del training.")
    return saved


def in_air_pressure(stats) -> float:
    """
    Valore normalizzato della pressione 0, cioè quello che hanno gli input reali a penna sollevata.
    """
    return float((0.0 - stats.mean[dp.PRESSURE]) / stats.std[dp.PRESSURE])


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# Controlli e salvataggio
# =============================================================================

def output_paths(out_dir: str, model: str, fold: int, tag: str = "") -> tuple[Path, Path]:
    folder = Path(out_dir) / model
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"fold{fold}{tag}.npz", folder / f"fold{fold}{tag}.json"


def summarize(features: list[np.ndarray], labels: list[int]) -> dict:
    """Alcune statistiche per classe in unità originali, da confrontare con i dati reali."""
    out = {}
    for label in (0, 1):
        xs = [f for f, y in zip(features, labels) if y == label]
        if not xs:
            continue
        cat = np.concatenate(xs)
        down = cat[:, dp.PEN] > 0.5
        out[CLASS_NAME[label]] = {
            "recordings": len(xs),
            "mean_length": round(float(np.mean([len(x) for x in xs])), 1),
            "pen_down_share": round(float(down.mean()), 3),
            "median_abs_offset_on_surface": round(float(np.median(np.abs(cat[down][:, :2]))), 3) if down.any() else None,
            "mean_pressure_on_surface": round(float(cat[down, dp.PRESSURE].mean()), 1) if down.any() else None,
        }
    return out


def check_and_save(features_raw: list[np.ndarray], plan: list[dict], npz_path: Path, json_path: Path,
                   meta: dict, recordings: list[dict], split: dict) -> dict:
    """
    Controlla le registrazioni generate, scrive .npz e .json, rilegge il .npz e lo confronta.

    Restituisce il riepilogo.
    """
    problems = []
    for f, p in zip(features_raw, plan):
        if f.shape != (p["length"], dp.N_CHANNELS):
            problems.append(f"{p['patient_id']} task {p['task']}: shape {f.shape}, attesa ({p['length']}, 6)")
        elif not np.isfinite(f).all():
            problems.append(f"{p['patient_id']} task {p['task']}: valori non finiti")
        elif not set(np.unique(f[:, dp.PEN])) <= {0.0, 1.0}:
            problems.append(f"{p['patient_id']} task {p['task']}: pen non è 0/1")
    if problems:
        raise SystemExit("I dati generati non hanno superato i controlli; non è stato salvato nulla:\n   " + "\n   ".join(problems[:10]))

    synthetic = [{"patient_id": p["patient_id"], "label": p["label"], "task": p["task"],
                  "features": f.astype(np.float32), "time_gaps": 0} for f, p in zip(features_raw, plan)]
    temporary = npz_path.with_name(npz_path.stem + ".tmp.npz")
    save_packed(synthetic, temporary)
    back = load_packed(temporary)                                    # stessa funzione di lettura usata per i dati reali
    by_key = {(r["patient_id"], r["task"]): r["features"] for r in back}
    assert len(back) == len(synthetic) and all(np.array_equal(by_key[(s["patient_id"], s["task"])], s["features"])
                                               for s in synthetic), "il file riletto è diverso"
    os.replace(temporary, npz_path)

    train_ids = set(split["train"])
    real_train = [r for r in recordings if r["patient_id"] in train_ids]
    summary = {"synthetic": summarize(features_raw, [p["label"] for p in plan]),
               "real_training_recordings_of_this_fold": summarize([r["features"] for r in real_train],
                                                                  [r["label"] for r in real_train])}
    meta = {**meta, "n_recordings": len(plan), "summary": summary,
            "plan": [{k: p[k] for k in ("patient_id", "label", "task", "length")} for p in plan]}
    json_path.write_text(json.dumps(meta, indent=1))
    return summary


def print_summary(summary: dict) -> None:
    for source, per_class in summary.items():
        print(f"   {source}:")
        for name, s in per_class.items():
            print(f"      {name:3s} n={s['recordings']:4d}  lunghezza media {s['mean_length']:7.1f}  "
                  f"penna giù {100 * s['pen_down_share']:5.1f}%  mediana |offset| su superficie "
                  f"{s['median_abs_offset_on_surface']}  pressione media su superficie {s['mean_pressure_on_surface']}")


def add_common_arguments(parser) -> None:
    parser.add_argument("--folds", type=int, nargs="+", required=True, help="es. --folds 0 1 2 3")
    parser.add_argument("--runs_dir", type=str, required=True,
                        help="cartella con fold0/, fold1/, ... di questo modello (es. runs/mdn)")
    parser.add_argument("--out_dir", type=str, default="synthetic", help="i risultati vanno in <out_dir>/<modello>/")
    parser.add_argument("--data_file", type=str, default=None, help="file .npz su cui sono stati addestrati i modelli")
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--n_per_cell", type=int, default=None,
                        help="registrazioni per (classe, task); default: come il dataset reale (36)")
    parser.add_argument("--seed", type=int, default=2026, help="seed del piano delle lunghezze e del campionamento")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="sovrascrive i file già presenti")
    parser.add_argument("--smoke", action="store_true",
                        help="2 registrazioni per (classe, task), scritte in <out_dir>_smoke")


# =============================================================================
# Memoria GPU esaurita
# =============================================================================

OutOfMemory = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


def generate_in_pieces(fn, group: list[int], log: list) -> list:
    """
    fn(gruppo) -> una registrazione generata per ogni elemento del gruppo.

    Se la GPU esaurisce la memoria, il batch viene diviso in due e ogni metà riprovata,
    fino alle singole registrazioni. Le divisioni sono annotate in `log` (salvato nel .json).
    Se non entra nemmeno una registrazione da sola, si ferma con un messaggio.
    """
    try:
        return fn(group)
    except OutOfMemory:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(group) == 1:
            raise SystemExit("Una sola registrazione non entra nella memoria della GPU. Generare con questo modello "
                             "su una GPU non condivisa con altri processi.")
        log.append(len(group))
        half = len(group) // 2
        return generate_in_pieces(fn, group[:half], log) + generate_in_pieces(fn, group[half:], log)
