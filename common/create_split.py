"""
create_split.py: split a livello di paziente per la cross-validation a 4 fold.

Stratificato per classe. Ogni fold: 18 pazienti di test (9 sani, 9 PD); dei 54
restanti, 8 di validazione (4 + 4) e 46 di training. Ogni paziente è in test una
sola volta. Il file viene creato una volta sola (seed 42) e non si sovrascrive
senza --force.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import dataset_prep as dp

SPLIT_FILE = Path("Cache/splits_4fold.json")
N_FOLDS = 4
VAL_PER_CLASS = 4
SEED = 42
CLASS_NAMES = {0: "Healthy", 1: "PD"}


def find_valid_patients() -> tuple[dict[str, int], list[dict]]:
    """Pazienti che dataset_prep riesce a caricare, con la loro etichetta."""
    labels = dp.load_labels(dp.LABELS_CSV)
    recordings, excluded = dp.load_recordings(dp.RAW_DATA_DIR, labels)
    patients = {r["patient_id"]: r["label"] for r in recordings}

    print(f"Pazienti validi: {len(patients)}   (esclusi: {len(excluded)})")
    for reason in excluded:
        print(f"   - {reason}")
    return patients, recordings


def build_folds(patients: dict[str, int], n_folds: int = N_FOLDS,
                val_per_class: int = VAL_PER_CLASS, seed: int = SEED) -> list[dict]:
    """Costruisce i fold stratificati per classe."""
    rng = random.Random(seed)

    groups_per_class: dict[int, list[list[str]]] = {}
    for label in sorted(set(patients.values())):
        ids = sorted(pid for pid, lab in patients.items() if lab == label)
        rng.shuffle(ids)
        groups_per_class[label] = [ids[k::n_folds] for k in range(n_folds)]

    folds = []
    for k in range(n_folds):
        train, val, test = [], [], []
        for label, groups in groups_per_class.items():
            test += groups[k]

            rest = [pid for j, group in enumerate(groups) if j != k for pid in group]
            rest = sorted(rest)
            random.Random(f"{seed}-val-{k}-{label}").shuffle(rest)
            val += rest[:val_per_class]
            train += rest[val_per_class:]

        folds.append({"fold": k, "train": sorted(train), "val": sorted(val), "test": sorted(test)})
    return folds


def check_folds(folds: list[dict], patients: dict[str, int]) -> None:
    """
    Controlla che train, val e test non si sovrappongano e che ogni paziente sia in test una volta.
    """
    everyone = set(patients)
    tested_so_far: set[str] = set()

    for f in folds:
        train, val, test = set(f["train"]), set(f["val"]), set(f["test"])

        assert not (train & val) and not (train & test) and not (val & test), f"fold {f['fold']}: sovrapposizione rilevata"
        assert train | val | test == everyone, f"fold {f['fold']}: pazienti mancanti"
        assert not (test & tested_so_far), f"fold {f['fold']}: paziente in test riutilizzato"
        tested_so_far |= test

        for name, part in (("train", train), ("val", val), ("test", test)):
            counts = np.bincount([patients[p] for p in part], minlength=2)
            if counts[0] != counts[1]:
                print(f"   WARNING fold {f['fold']} {name}: {counts[0]} Sani vs {counts[1]} PD")

    assert tested_so_far == everyone, "Almeno un paziente non è stato allocato in nessun test set"


def save_split(folds: list[dict], patients: dict[str, int], path: Path = SPLIT_FILE) -> None:
    """Salva lo split in JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "description": "Patient-level stratified K-fold split for the PaHaW generative benchmark",
            "seed": SEED,
            "n_folds": len(folds),
            "val_per_class": VAL_PER_CLASS,
            "labels": {pid: patients[pid] for pid in sorted(patients)},
            "folds": folds,
        }, f, indent=2)


def load_fold(fold: int, path: Path = SPLIT_FILE) -> dict:
    """Legge lo split e restituisce il fold richiesto."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} non trovato. Crearlo con: python common/create_split.py")
    with open(path) as f:
        split = json.load(f)
    if not 0 <= fold < split["n_folds"]:
        raise ValueError(f"Il fold deve essere tra 0 e {split['n_folds'] - 1}")
    return split["folds"][fold]


def print_summary(folds: list[dict], patients: dict[str, int], recordings: list[dict]) -> None:
    """Stampa per ogni fold il numero di pazienti per parte e classe e la lunghezza media."""
    lengths: dict[str, list[int]] = {}
    for r in recordings:
        lengths.setdefault(r["patient_id"], []).append(len(r["features"]))

    print(f"\n{'fold':>4} {'part':>6} {'pazienti':>9} {'Sani':>8} {'PD':>4} {'lunghezza media':>15}")
    for f in folds:
        for name in ("train", "val", "test"):
            ids = f[name]
            n_pd = sum(patients[p] for p in ids)
            mean_len = np.mean([n for p in ids for n in lengths[p]])
            print(f"{f['fold']:>4} {name:>6} {len(ids):>9} {len(ids) - n_pd:>8} {n_pd:>4} {mean_len:>15.0f}")


def main() -> None:
    """Crea lo split, lo controlla e lo salva."""
    parser = argparse.ArgumentParser(description="Crea lo split a 4 fold per paziente.")
    parser.add_argument("--force", action="store_true",
                        help="sovrascrive lo split esistente (i modelli già addestrati non sono più confrontabili)")
    args = parser.parse_args()

    if SPLIT_FILE.exists() and not args.force:
        raise SystemExit(
            f"{SPLIT_FILE} esiste già e non è stato modificato: tutti gli esperimenti devono usare lo stesso split.\n"
            f"Per crearne uno nuovo (e riaddestrare tutto): python common/create_split.py --force"
        )

    patients, recordings = find_valid_patients()
    folds = build_folds(patients)
    check_folds(folds, patients)
    save_split(folds, patients)

    print_summary(folds, patients, recordings)
    print(f"\nSalvato in {SPLIT_FILE} (seed {SEED})")


if __name__ == "__main__":
    main()