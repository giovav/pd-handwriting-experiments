"""
packed_data.py: salva tutte le registrazioni elaborate in un unico file .npz e le rilegge
nello stesso formato di dataset_prep.load_recordings (per fare il training senza i .svc).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import dataset_prep as dp

FORMAT_VERSION = 2


def save_packed(recordings: list[dict], path: Path) -> None:
    """Scrive le registrazioni in un .npz compresso, nell'ordine in cui sono date."""
    patient_ids = sorted({r["patient_id"] for r in recordings})
    index_of = {pid: i for i, pid in enumerate(patient_ids)}

    meta = {
        "format_version": FORMAT_VERSION,
        "channel_names": dp.CHANNEL_NAMES,
        "patient_ids": patient_ids,
        "n_recordings": len(recordings),
        "gap_settings": dp.gap_settings(),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.concatenate([r["features"] for r in recordings], axis=0).astype(np.float32),
        lengths=np.array([len(r["features"]) for r in recordings], dtype=np.int32),
        labels=np.array([r["label"] for r in recordings], dtype=np.int8),
        tasks=np.array([r["task"] for r in recordings], dtype=np.int8),
        patient_index=np.array([index_of[r["patient_id"]] for r in recordings], dtype=np.int16),
        time_gaps=np.array([r["time_gaps"] for r in recordings], dtype=np.int32),
        meta=np.array(json.dumps(meta)),
    )


def read_meta(path: Path) -> dict:
    """Legge solo i metadati del file."""
    with np.load(Path(path), allow_pickle=False) as data:
        return json.loads(str(data["meta"]))


def load_packed(path: Path, strict: bool = True) -> list[dict]:
    """
    Legge un .npz scritto da save_packed e restituisce la lista di registrazioni.

    Con strict=True rifiuta i file con versione del formato o regole sui gap diverse
    da quelle del codice attuale.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        features, lengths = data["features"], data["lengths"]
        labels, tasks = data["labels"], data["tasks"]
        patient_index, time_gaps = data["patient_index"], data["time_gaps"]

    if strict:
        made_with = meta.get("gap_settings", {}).get("rules", "the rules before 2026-10-07")
        if meta["format_version"] != FORMAT_VERSION or made_with != dp.GAP_RULES:
            raise ValueError(
                f"{path} è stato esportato con regole diverse da quelle del codice.\n"
                f"   regole del file  : {made_with}\n"
                f"   regole del codice: {dp.GAP_RULES}\n"
                f"Rigenerarlo con export_packed_data.py."
            )
    if meta["channel_names"] != dp.CHANNEL_NAMES:
        raise ValueError(f"{path}: canali {meta['channel_names']} diversi da quelli di dataset_prep.py {dp.CHANNEL_NAMES}")
    if int(lengths.sum()) != features.shape[0] or features.shape[1] != dp.N_CHANNELS:
        raise ValueError(f"{path}: file incoerente (upload interrotto?).")

    ends = np.cumsum(lengths)
    starts = ends - lengths
    patient_ids = meta["patient_ids"]

    recordings = []
    for i in range(len(lengths)):
        recordings.append({
            "patient_id": patient_ids[int(patient_index[i])],
            "label": int(labels[i]),
            "task": int(tasks[i]),
            "features": features[starts[i]:ends[i]].copy(),
            "time_gaps": int(time_gaps[i]),
        })
    return recordings