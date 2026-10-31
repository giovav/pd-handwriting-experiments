"""
export_packed_data.py: prepara i dati per il training in un unico file, senza i .svc.

Sostituisce gli ID dei pazienti con pseudonimi (P000-P071, in ordine casuale con seed
fisso), tiene solo i 6 canali, classe e task, e scrive:
    packed_export/pahaw_packed.npz   registrazioni elaborate
    packed_export/splits_4fold.json  lo split con gli pseudonimi
La corrispondenza pseudonimo -> ID reale resta solo in Cache/id_map.json, che non va
mai condiviso. È una pseudonimizzazione, non un'anonimizzazione: il file .npz va
trattato con le stesse restrizioni dei dati originali.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

# I moduli condivisi stanno in common/ (se c'è una copia accanto allo script, vale quella).
sys.path.insert(1, str(Path(__file__).resolve().parent / "common"))

import dataset_prep as dp
from create_split import SPLIT_FILE
from packed_data import load_packed, save_packed

EXPORT_DIR = Path("packed_export")
PACKED_FILE = EXPORT_DIR / "pahaw_packed.npz"
EXPORTED_SPLIT_FILE = EXPORT_DIR / "splits_4fold.json"
ID_MAP_FILE = Path("Cache/id_map.json")

ID_SEED = 2026


def make_id_map(real_ids: list[str]) -> dict[str, str]:
    """Assegna gli pseudonimi P000, P001, ... in ordine casuale (seed fisso)."""
    shuffled = sorted(real_ids)
    random.Random(ID_SEED).shuffle(shuffled)
    return {real: f"P{i:03d}" for i, real in enumerate(shuffled)}


def translate_split(split: dict, id_map: dict[str, str]) -> dict:
    """Lo split con gli pseudonimi al posto degli ID originali."""
    return {
        "description": split["description"] + " (pseudonymous patient IDs)",
        "seed": split["seed"],
        "n_folds": split["n_folds"],
        "val_per_class": split["val_per_class"],
        "labels": dict(sorted((id_map[pid], label) for pid, label in split["labels"].items())),
        "folds": [
            {"fold": f["fold"], **{part: sorted(id_map[p] for p in f[part]) for part in ("train", "val", "test")}}
            for f in split["folds"]
        ],
    }


def verify_export(original: list[dict], id_map: dict[str, str], split: dict, new_split: dict) -> None:
    """Rilegge i file esportati e controlla che coincidano con i dati elaborati in locale."""
    packed = load_packed(PACKED_FILE)
    assert len(packed) == len(original), "numero di registrazioni diverso"
    
    by_key = {(r["patient_id"], r["task"]): r for r in packed}
    assert len(by_key) == len(packed), "coppie (paziente, task) duplicate nel file esportato"
    for a in original:
        b = by_key[(id_map[a["patient_id"]], a["task"])]
        assert a["label"] == b["label"]
        assert a["features"].dtype == b["features"].dtype == np.float32
        assert np.array_equal(a["features"], b["features"]), "registrazione diversa nel file esportato"

    for fold_raw, fold_new in zip(split["folds"], new_split["folds"]):
        stats_raw = dp.fit_norm_stats(original, fold_raw["train"])
        stats_new = dp.fit_norm_stats(packed, fold_new["train"])
        assert np.array_equal(stats_raw.mean, stats_new.mean) and np.array_equal(stats_raw.std, stats_new.std)
        for part in ("train", "val", "test"):
            assert {id_map[p] for p in fold_raw[part]} == set(fold_new[part])


def sha256(path: Path) -> str:
    """SHA-256 di un file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Esporta, rilegge e verifica."""
    if not SPLIT_FILE.exists():
        raise SystemExit(f"{SPLIT_FILE} non trovato. Crearlo prima con: python common/create_split.py")
    with open(SPLIT_FILE) as f:
        split = json.load(f)

    labels = dp.load_labels(dp.LABELS_CSV)
    recordings, _ = dp.load_recordings(dp.RAW_DATA_DIR, labels)
    real_ids = sorted({r["patient_id"] for r in recordings})
    if set(real_ids) != set(split["labels"]):
        raise SystemExit("I pazienti caricati non corrispondono a quelli dello split.")

    id_map = make_id_map(real_ids)
    renamed = [{**r, "patient_id": id_map[r["patient_id"]]} for r in recordings]
    renamed.sort(key=lambda r: (r["patient_id"], r["task"]))
    new_split = translate_split(split, id_map)

    save_packed(renamed, PACKED_FILE)
    with open(EXPORTED_SPLIT_FILE, "w") as f:
        json.dump(new_split, f, indent=2)

    ID_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ID_MAP_FILE, "w") as f:
        json.dump({"new_to_original": {new: real for real, new in sorted(id_map.items(), key=lambda kv: kv[1])}},
                  f, indent=2)

    verify_export(recordings, id_map, split, new_split)

    n_steps = sum(len(r["features"]) for r in recordings)
    n_pd = sum(split["labels"].values())
    print(f"Pazienti      : {len(real_ids)}  ({len(real_ids) - n_pd} sani, {n_pd} PD)")
    print(f"Registrazioni : {len(recordings)}   campioni in totale: {n_steps:,}")
    print(f"Regole sui gap: {dp.gap_settings()['rules']}   TRIM_AIR_EDGES = {dp.TRIM_AIR_EDGES}   "
          f"({sum(r['time_gaps'] for r in recordings):,} offset corretti)")
    print("Verifica superata: i file esportati contengono gli stessi dati.")
    print("\nFile per il training:")
    for path in (PACKED_FILE, EXPORTED_SPLIT_FILE):
        print(f"   {str(path):38s} {path.stat().st_size / 1e6:7.2f} MB   sha256 {sha256(path)[:16]}...")
    print(f"\nDa NON condividere né mettere su git:\n   {ID_MAP_FILE}")


if __name__ == "__main__":
    main()