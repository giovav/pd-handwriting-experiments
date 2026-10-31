"""
dataset_prep.py: lettura dei file .svc di PaHaW, estrazione dei 6 canali,
normalizzazione e DataLoader comuni a tutti i modelli.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

RAW_DATA_DIR = Path("PaHaW_public")
LABELS_CSV = Path("PaHaW_files/corpus_PaHaW.csv")
NORM_STATS_FILE = Path("Cache/norm_stats.json")
SVC_COLUMNS = ["x", "y", "time", "pen", "azimuth", "altitude", "pressure"]

TRAIN_PATIENT_IDS: list[str] = []

GAP_HANDLING = "rescale"
CLOCK_RESET_PERIODS = 10_000
TRIM_AIR_EDGES = False
N_TASKS = 8
REQUIRE_ALL_TASKS = True
MAX_SEQ_LEN: int | None = None
PAD_MULTIPLE = 8

CHANNEL_NAMES = ["dx", "dy", "pressure", "azimuth", "altitude", "pen"]
N_CHANNELS = len(CHANNEL_NAMES)
N_CONTINUOUS = N_CHANNELS - 1
DX, DY, PRESSURE, AZIMUTH, ALTITUDE, PEN = range(N_CHANNELS)

LABEL_MAP = {"H": 0, "HC": 0, "PD": 1}
GAP_RULES = "in the air only, 2026-10-07"
_SVC_NAME = re.compile(r"^(\d+)__(\d+)_(\d+)\.svc$")


def load_labels(csv_path: Path) -> dict[str, int]:
    """Legge il CSV delle etichette: ID paziente (5 cifre) -> 0 sano, 1 PD."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File etichette non trovato: {csv_path}")

    table = pd.read_csv(csv_path)
    labels: dict[str, int] = {}
    for _, row in table.iterrows():
        patient_id = str(row["ID"]).strip().zfill(5)
        disease = str(row["Disease"]).strip().upper()
        if disease not in LABEL_MAP:
            raise ValueError(f"Valore Disease sconosciuto '{disease}' per {patient_id}")
        labels[patient_id] = LABEL_MAP[disease]
    return labels


def parse_svc_file(path: Path) -> np.ndarray:
    """Legge un file .svc (la prima riga è il numero di campioni) in un array [N, 7]."""
    with open(path) as f:
        rows = f.read().split("\n")[1:]
    rows = [r.split() for r in rows if r.strip()]
    if not rows:
        return np.zeros((0, len(SVC_COLUMNS)), dtype=np.float64)

    raw = np.array(rows, dtype=np.float64)
    if raw.shape[1] != len(SVC_COLUMNS):
        raise ValueError(f"{path}: attese {len(SVC_COLUMNS)} colonne, trovate {raw.shape[1]}")
    return raw


def find_task_files(patient_dir: Path) -> dict[int, Path]:
    """File .svc di un paziente, indicizzati per numero di task."""
    files: dict[int, Path] = {}
    for path in sorted(patient_dir.glob("*.svc")):
        match = _SVC_NAME.match(path.name)
        if match:
            files[int(match.group(2))] = path
    return files


def gap_settings() -> dict:
    """Regole sui gap usate, salvate nei metadati del file packed."""
    return {
        "rules": GAP_RULES if GAP_HANDLING == "rescale" else "none",
        "trim_air_edges": bool(TRIM_AIR_EDGES)
    }


def time_steps(raw: np.ndarray) -> np.ndarray:
    """Passo temporale di ogni campione in periodi di campionamento (mediana = 1)."""
    time = raw[:, SVC_COLUMNS.index("time")]
    steps = np.ones(len(time), dtype=np.float64)
    if len(time) < 3:
        return steps
    differences = np.diff(time)
    typical = np.median(differences)
    if typical > 0:
        steps[1:] = differences / typical
    return steps


def elapsed_periods(raw: np.ndarray) -> np.ndarray:
    """Periodi interi trascorsi tra un campione e il precedente (1 se non c'è gap)."""
    steps = time_steps(raw)
    periods = np.ones(len(steps), dtype=np.int64)
    is_gap = steps > 1.5
    periods[is_gap] = np.round(steps[is_gap]).astype(np.int64)
    return periods


def pen_in_air(raw: np.ndarray) -> np.ndarray:
    """True dove la penna è sollevata nel campione corrente o nel precedente."""
    pen = raw[:, SVC_COLUMNS.index("pen")] > 0
    pen_before = np.concatenate([pen[:1], pen[:-1]])
    return ~(pen & pen_before)


def gap_correction(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divisore per dx, dy (periodi trascorsi, solo in aria) e posizioni con reset del clock."""
    steps = time_steps(raw)
    in_air = pen_in_air(raw)
    divisor = np.where(in_air, elapsed_periods(raw), 1)
    is_reset = in_air & (np.abs(steps) > CLOCK_RESET_PERIODS)
    divisor[is_reset] = 1
    return divisor, is_reset


def extract_features(raw: np.ndarray) -> np.ndarray:
    """Matrice grezza [N, 7] -> canali [dx, dy, pressure, azimuth, altitude, pen]."""
    column = {name: raw[:, i] for i, name in enumerate(SVC_COLUMNS)}

    dx = np.diff(column["x"], prepend=column["x"][0])
    dy = np.diff(column["y"], prepend=column["y"][0])

    if GAP_HANDLING == "rescale":
        divisor, is_reset = gap_correction(raw)
        dx = np.where(is_reset, 0.0, dx / divisor)
        dy = np.where(is_reset, 0.0, dy / divisor)
    elif GAP_HANDLING != "none":
        raise ValueError(f"GAP_HANDLING non valido: '{GAP_HANDLING}'")

    pen = (column["pen"] > 0).astype(np.float64)
    pressure = column["pressure"] * pen

    features = np.stack(
        [dx, dy, pressure, column["azimuth"], column["altitude"], pen], axis=1
    )
    return features.astype(np.float32)


def offsets_changed(raw: np.ndarray) -> np.ndarray:
    """True dove la correzione dei gap ha modificato dx, dy."""
    if GAP_HANDLING != "rescale":
        return np.zeros(len(raw), dtype=bool)
    divisor, is_reset = gap_correction(raw)
    return (divisor > 1) | is_reset


def count_time_gaps(raw: np.ndarray) -> int:
    """Numero di offset modificati dalla correzione dei gap."""
    return int(offsets_changed(raw).sum())


def task_span(raw: np.ndarray) -> tuple[int, int]:
    """Indici del primo e dell'ultimo campione con la penna sulla superficie."""
    on_surface = np.where(raw[:, SVC_COLUMNS.index("pen")] > 0)[0]
    if len(on_surface) == 0:
        return 0, len(raw)
    return int(on_surface[0]), int(on_surface[-1]) + 1


def load_recordings(raw_dir: Path, labels: dict[str, int]) -> tuple[list[dict], list[str]]:
    """Carica tutte le registrazioni; esclude i pazienti senza tutti gli 8 task."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Cartella dati non trovata: {raw_dir}")

    recordings: list[dict] = []
    excluded: list[str] = []

    for patient_id, label in sorted(labels.items()):
        patient_dir = raw_dir / patient_id
        if not patient_dir.is_dir():
            excluded.append(f"{patient_id}: cartella non trovata")
            continue

        patient_recordings = []
        for task_number, path in find_task_files(patient_dir).items():
            raw = parse_svc_file(path)
            if len(raw) < 2 or not 1 <= task_number <= N_TASKS:
                continue

            features = extract_features(raw)
            changed = offsets_changed(raw)
            if TRIM_AIR_EDGES:
                start, stop = task_span(raw)
                features = features[start:stop].copy()
                features[0, :2] = 0.0
                changed = changed[start + 1:stop]
            if len(features) < 2:
                continue

            patient_recordings.append({
                "patient_id": patient_id,
                "label": label,
                "task": task_number - 1,
                "features": features,
                "time_gaps": int(changed.sum()),
            })

        if REQUIRE_ALL_TASKS and len(patient_recordings) != N_TASKS:
            excluded.append(f"{patient_id}: {len(patient_recordings)} task validi su {N_TASKS}")
            continue
        recordings.extend(patient_recordings)

    return recordings, excluded


@dataclass
class NormStats:
    """Medie e deviazioni standard dei canali continui, con i pazienti usati per calcolarle."""
    mean: np.ndarray
    std: np.ndarray
    train_patient_ids: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "channels": CHANNEL_NAMES[:N_CONTINUOUS],
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "train_patient_ids": self.train_patient_ids,
            }, f, indent=2)

    @staticmethod
    def load(path: Path) -> "NormStats":
        with open(path) as f:
            d = json.load(f)
        return NormStats(
            mean=np.array(d["mean"], dtype=np.float32),
            std=np.array(d["std"], dtype=np.float32),
            train_patient_ids=d["train_patient_ids"],
        )


def fit_norm_stats(recordings: list[dict], train_patient_ids: list[str]) -> NormStats:
    """
    Statistiche calcolate solo sui pazienti di training.

    dx, dy: media 0 e una scala comune (RMS dei due canali); pressione: solo sui
    campioni con penna appoggiata; azimuth e altitude: media e deviazione standard.
    """
    train_ids = set(train_patient_ids)
    if not train_ids:
        raise ValueError("TRAIN_PATIENT_IDS è vuota: servono gli ID di training.")

    available = {r["patient_id"] for r in recordings}
    unknown = sorted(train_ids - available)
    if unknown:
        raise ValueError(f"ID di training senza registrazioni caricate: {unknown}")

    data = np.concatenate(
        [r["features"] for r in recordings if r["patient_id"] in train_ids], axis=0
    ).astype(np.float64)

    mean = np.zeros(N_CONTINUOUS, dtype=np.float64)
    std = np.ones(N_CONTINUOUS, dtype=np.float64)

    shared_scale = np.sqrt((np.mean(data[:, DX] ** 2) + np.mean(data[:, DY] ** 2)) / 2)
    std[DX] = std[DY] = shared_scale

    on_surface = data[:, PEN] == 1
    if on_surface.any():
        mean[PRESSURE] = data[on_surface, PRESSURE].mean()
        std[PRESSURE] = data[on_surface, PRESSURE].std()

    for ch in (AZIMUTH, ALTITUDE):
        mean[ch] = data[:, ch].mean()
        std[ch] = data[:, ch].std()

    std[std < 1e-8] = 1.0

    return NormStats(mean.astype(np.float32), std.astype(np.float32), sorted(train_ids))


def normalize(features: np.ndarray, stats: NormStats) -> np.ndarray:
    """Normalizza i canali continui; pen resta 0/1."""
    out = features.astype(np.float32).copy()
    out[..., :N_CONTINUOUS] = (out[..., :N_CONTINUOUS] - stats.mean) / stats.std
    return out


def denormalize(features: np.ndarray, stats: NormStats) -> np.ndarray:
    """Torna alle unità originali; pen viene binarizzato e la pressione azzerata in aria."""
    out = features.astype(np.float32).copy()
    out[..., :N_CONTINUOUS] = out[..., :N_CONTINUOUS] * stats.std + stats.mean
    out[..., PEN] = (out[..., PEN] > 0.5).astype(np.float32)
    out[..., PRESSURE] *= out[..., PEN]
    return out


def offsets_to_trajectory(raw_features: np.ndarray) -> np.ndarray:
    """Traiettoria assoluta (x, y) come somma cumulativa degli offset."""
    return np.cumsum(raw_features[:, [DX, DY]], axis=0)


class PaHaWSequenceDataset(Dataset):
    """Una registrazione completa per elemento, già normalizzata."""
    
    def __init__(
        self,
        recordings: list[dict],
        stats: NormStats,
        patient_ids: list[str] | None = None,
        max_len: int | None = MAX_SEQ_LEN,
    ):
        keep = set(patient_ids) if patient_ids is not None else None

        self.items: list[dict] = []
        self.n_dropped_too_long = 0

        for rec in recordings:
            if keep is not None and rec["patient_id"] not in keep:
                continue
            if max_len is not None and len(rec["features"]) > max_len:
                self.n_dropped_too_long += 1
                continue

            self.items.append({
                "x": torch.from_numpy(normalize(rec["features"], stats)),
                "y": rec["label"],
                "task": rec["task"],
                "patient_id": rec["patient_id"],
            })

        self.lengths = [item["x"].shape[0] for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def collate_fn(batch: list[dict], pad_multiple: int = PAD_MULTIPLE) -> dict:
    """Crea il batch con padding a un multiplo di 8, maschera (True = dato reale) e lunghezze."""
    batch_size = len(batch)
    lengths = torch.tensor([item["x"].shape[0] for item in batch], dtype=torch.int64)
    padded_len = math.ceil(int(lengths.max()) / pad_multiple) * pad_multiple

    x = torch.zeros(batch_size, padded_len, N_CHANNELS, dtype=torch.float32)
    for i, item in enumerate(batch):
        x[i, : lengths[i]] = item["x"]

    mask = torch.arange(padded_len)[None, :] < lengths[:, None]

    return {
        "x": x,
        "mask": mask,
        "lengths": lengths,
        "y": torch.tensor([item["y"] for item in batch], dtype=torch.int64),
        "task": torch.tensor([item["task"] for item in batch], dtype=torch.int64),
        "patient_id": [item["patient_id"] for item in batch],
    }


class BucketBatchSampler(Sampler):
    """Batch di registrazioni di lunghezza simile, per ridurre il padding."""

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True,
                 bucket_factor: int = 20, seed: int | None = None):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_factor = bucket_factor
        self.rng = random.Random(seed)

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            self.rng.shuffle(indices)
            bucket_size = self.batch_size * self.bucket_factor
        else:
            bucket_size = len(indices)

        batches = []
        for start in range(0, len(indices), max(bucket_size, 1)):
            bucket = sorted(indices[start:start + bucket_size], key=lambda i: self.lengths[i])
            for b in range(0, len(bucket), self.batch_size):
                batches.append(bucket[b:b + self.batch_size])

        if self.shuffle:
            self.rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


def make_dataloader(dataset: PaHaWSequenceDataset, batch_size: int, shuffle: bool,
                    bucket_by_length: bool = True, num_workers: int = 0) -> DataLoader:
    """DataLoader con BucketBatchSampler e collate_fn."""
    if bucket_by_length:
        sampler = BucketBatchSampler(dataset.lengths, batch_size, shuffle=shuffle)
        return DataLoader(dataset, batch_sampler=sampler,
                          collate_fn=collate_fn, num_workers=num_workers)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=num_workers)


def print_summary(recordings: list[dict], excluded: list[str]) -> None:
    """Stampa il riepilogo del caricamento."""
    patients = {r["patient_id"]: r["label"] for r in recordings}
    n_pd = sum(patients.values())
    lengths = np.array([len(r["features"]) for r in recordings])

    print(f"\nPazienti caricati : {len(patients)}  ({len(patients) - n_pd} Sani, {n_pd} PD)")
    print(f"Registrazioni     : {len(recordings)}")
    print(f"Pazienti esclusi  : {len(excluded)}")
    for reason in excluded:
        print(f"   - {reason}")

    print("\nLunghezza registrazioni (time steps):")
    print(f"   min {lengths.min()}, media {lengths.mean():.0f}, max {lengths.max()}")
    for p in (50, 90, 95, 99):
        print(f"   {p}° percentile: {np.percentile(lengths, p):.0f}")
    for label, name in ((0, "Sani"), (1, "PD")):
        class_lengths = lengths[[r["label"] == label for r in recordings]]
        if len(class_lengths):
            print(f"   media lunghezza {name}: {class_lengths.mean():.0f}")

    with_gaps = sum(r["time_gaps"] > 0 for r in recordings)
    n_gaps = sum(r["time_gaps"] for r in recordings)
    print(f"\nGAP_HANDLING = '{GAP_HANDLING}', TRIM_AIR_EDGES = {TRIM_AIR_EDGES}")
    print(f"Offset corretti su gap in aria: {n_gaps} in {with_gaps} di {len(recordings)} totali")


def main(raw_dir: Path = RAW_DATA_DIR, labels_csv: Path = LABELS_CSV,
         train_patient_ids: list[str] | None = None) -> None:
    """Controllo rapido: carica i dati, calcola le statistiche e prova un batch."""
    train_ids = TRAIN_PATIENT_IDS if train_patient_ids is None else train_patient_ids

    labels = load_labels(labels_csv)
    recordings, excluded = load_recordings(raw_dir, labels)
    if not recordings:
        raise RuntimeError("Nessuna registrazione caricata. Controllare RAW_DATA_DIR e LABELS_CSV.")
    print_summary(recordings, excluded)

    smoke_test = len(train_ids) == 0
    if smoke_test:
        print("\n" + "!" * 78)
        print("ATTENZIONE: TRAIN_PATIENT_IDS vuoto. SMOKE TEST IN CORSO.")
        print("Le statistiche sono calcolate su tutti i pazienti e NON vengono salvate.")
        print("!" * 78)
        train_ids = sorted({r["patient_id"] for r in recordings})

    stats = fit_norm_stats(recordings, train_ids)
    print("\nCostanti di normalizzazione:")
    for name, m, s in zip(CHANNEL_NAMES, stats.mean, stats.std):
        print(f"   {name:9s} media {m:10.3f}   std {s:10.3f}")
    if not smoke_test:
        stats.save(NORM_STATS_FILE)
        print(f"Salvate in {NORM_STATS_FILE}")

    dataset = PaHaWSequenceDataset(recordings, stats, patient_ids=train_ids)
    if dataset.n_dropped_too_long:
        print(f"\nScartate {dataset.n_dropped_too_long} registrazioni che superano MAX_SEQ_LEN")
    loader = make_dataloader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))

    print("\nDettaglio campione batch:")
    for key in ("x", "mask", "lengths", "y", "task"):
        print(f"   {key:8s} shape {str(tuple(batch[key].shape)):16s} dtype {batch[key].dtype}")
    padding = 1 - batch["mask"].float().mean().item()
    print(f"   rapporto di padding batch: {100 * padding:.1f}%")

    assert batch["x"].shape[1] % PAD_MULTIPLE == 0
    assert batch["x"].shape[2] == N_CHANNELS
    assert torch.equal(batch["mask"].sum(dim=1), batch["lengths"])
    assert (batch["x"][~batch["mask"]] == 0).all()
    assert set(batch["x"][..., PEN].unique().tolist()) <= {0.0, 1.0}
    assert batch["y"].dtype == torch.int64 and batch["task"].dtype == torch.int64
    print("\nTutti i test superati.")


if __name__ == "__main__":
    main()