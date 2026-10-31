"""
check_jumps.py: controllo, sui dati grezzi, delle regole sui gap di dataset_prep.py.
Solo lettura; da lanciare in locale dalla cartella principale.

Stampa sei parti:
    1. passi indietro del clock e reset;
    2. test del rapporto ai gap, su superficie e in aria;
    3. offset grezzi, divisi a ogni gap e con le regole finali, a confronto;
    4. offset rimasti oltre 20 deviazioni standard, raggruppati in eventi e classificati;
    5. cosa toglierebbe TRIM_AIR_EDGES;
    6. controllo che dataset_prep.py applichi esattamente le regole e confronto con
       l'ultimo export in packed_export/.
Alla fine dà un verdetto.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# I moduli condivisi stanno in common/ (se c'è una copia accanto allo script, vale quella).
sys.path.insert(1, str(Path(__file__).resolve().parent / "common"))

import dataset_prep as dp

BIG = 20.0
LIMIT = 60.0


def load_rows():
    """
    Per ogni registrazione: offset grezzi, divisi a ogni gap e con le regole finali, più le maschere usate.
    """
    labels = dp.load_labels(dp.LABELS_CSV)
    recordings, _ = dp.load_recordings(dp.RAW_DATA_DIR, labels)
    valid = sorted({r["patient_id"] for r in recordings})
    produced = {(r["patient_id"], r["task"]): r["features"] for r in recordings}

    col = {name: i for i, name in enumerate(dp.SVC_COLUMNS)}
    rows = []
    for pid in valid:
        for task_number, path in dp.find_task_files(dp.RAW_DATA_DIR / pid).items():
            raw = dp.parse_svc_file(path)
            pen = raw[:, col["pen"]] > 0
            divisor, is_reset = dp.gap_correction(raw)
            dx = np.diff(raw[:, col["x"]], prepend=raw[0, col["x"]])
            dy = np.diff(raw[:, col["y"]], prepend=raw[0, col["y"]])
            periods = dp.elapsed_periods(raw)
            rows.append({
                "pid": pid, "task": task_number, "label": labels[pid], "n": len(raw),
                "time": raw[:, col["time"]], "step": dp.time_steps(raw), "periods": periods,
                "pen": pen, "air": dp.pen_in_air(raw), "span": dp.task_span(raw),
                "dx": dx, "dy": dy,
                "ex": dx / periods, "ey": dy / periods,
                "fx": np.where(is_reset, 0.0, dx / divisor),
                "fy": np.where(is_reset, 0.0, dy / divisor),
                "divisor": divisor, "is_reset": is_reset,
                "produced": produced[(pid, task_number - 1)],
            })
    return rows


def read_old_export(path: Path):
    """Legge l'export precedente (packed_export/pahaw_packed.npz) per il confronto della parte 6."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        features, lengths = data["features"], data["lengths"]
        tasks, patient_index = data["tasks"], data["patient_index"]
    ends = np.cumsum(lengths)
    return [(meta["patient_ids"][int(patient_index[i])], int(tasks[i]), features[ends[i] - lengths[i]:ends[i]])
            for i in range(len(lengths))]


def scale_of(rows, a: str, b: str) -> float:
    """Scala comune di dx, dy (RMS) per la variante a, b degli offset."""
    x, y = np.concatenate([r[a] for r in rows]), np.concatenate([r[b] for r in rows])
    return float(np.sqrt((np.mean(x ** 2) + np.mean(y ** 2)) / 2))


def cat(rows, key: str) -> np.ndarray:
    return np.concatenate([r[key] for r in rows])


def high(values: np.ndarray, quantile: float = 1.0) -> float:
    """Quantile q dei valori (massimo se q = 1)."""
    return float(np.quantile(values, quantile)) if len(values) else float("nan")


def size_in_sigma(rows, a: str, b: str, scale: float) -> np.ndarray:
    """Lunghezza degli offset in deviazioni standard."""
    return np.concatenate([np.maximum(np.abs(r[a]), np.abs(r[b])) for r in rows]) / scale


def main() -> None:
    """Esegue i controlli e stampa il report."""
    rows = load_rows()
    checks = {}
    print(f"{len(rows)} registrazioni di {len({r['pid'] for r in rows})} pazienti   "
          f"GAP_HANDLING = '{dp.GAP_HANDLING}'   TRIM_AIR_EDGES = {dp.TRIM_AIR_EDGES}")
    
    if dp.GAP_HANDLING != "rescale":
        raise SystemExit("Per controllare le regole sui gap serve GAP_HANDLING = 'rescale' in dataset_prep.py.")
    if dp.TRIM_AIR_EDGES:
        print("NOTA: le parti 1-4 riguardano le registrazioni intere, prima del taglio; la parte 5 dice cosa toglie il taglio.")

    step, periods, air = cat(rows, "step"), cat(rows, "periods"), cat(rows, "air")
    n_all, n_air = len(step), int(air.sum())
    print(f"{n_all:,} campioni: {n_all - n_air:,} con la penna sulla superficie, {n_air:,} in aria "
          f"({100 * n_air / n_all:.1f}%)")

    # [1] clock: passi indietro e reset
    print("\nPARTE 1: clock")
    reset = np.abs(step) > dp.CLOCK_RESET_PERIODS
    back = (step <= 0) & ~reset
    print(f"   passi indietro o nulli: {int(back.sum())} "
          f"({int((back & ~air).sum())} su superficie, {int((back & air).sum())} in aria)")
    
    if back.any():
        print(f"      ampiezza in periodi: mediana {np.median(-step[back]):.1f}, "
              f"massimo {(-step[back]).max():.1f}")
    
    print(f"   reset (passi oltre {dp.CLOCK_RESET_PERIODS} periodi): {int(reset.sum())}")
    for r in rows:
        for i in np.where(np.abs(r["step"]) > dp.CLOCK_RESET_PERIODS)[0]:
            t = r["time"]
            around = ", ".join(f"{t[j]:.0f}" for j in range(max(0, i - 1), min(r["n"], i + 2)))
            print(f"      -> paziente {r['pid']} task {r['task']} campione {i}/{r['n']}  "
                  f"[{'aria' if r['air'][i] else 'SUPERFICIE'}]  timestamp vicini: {around}")
    
    for label, name in ((0, "Sani"), (1, "PD")):
        recorded = sum(r["n"] for r in rows if r["label"] == label)
        missed = sum(int((r["divisor"] - 1).sum()) for r in rows if r["label"] == label)
        print(f"   {name:8s}: periodi mancanti nei gap in aria = {100 * missed / recorded:5.1f}% dei campioni registrati")

    # [2] gap: il campione manca davvero?
    print("\nPARTE 2: test del rapporto ai gap (2.0 se manca un campione, 1.0 se il timestamp è solo in ritardo)")
    print(f"   {'gap':>16s} | {'SU SUPERFICIE':^28s} | {'IN ARIA':^28s}")
    print(f"   {'':>16s} | {'gaps':>8s} {'ratio':>8s} {'(cases)':>9s} | {'gaps':>8s} {'ratio':>8s} {'(cases)':>9s}")
    
    buckets = ((1, 1), (2, 2), (3, 3), (4, 7), (8, 30), (31, dp.CLOCK_RESET_PERIODS))
    table = {}
    for wanted_air in (False, True):
        ratios = {b: [] for b in buckets}
        for r in rows:
            mag, p, pn, n = np.hypot(r["dx"], r["dy"]), r["periods"], r["pen"], r["n"]
            if n < 12:
                continue
            idx = np.arange(3, n - 3)
            clean = (p[idx - 2] == 1) & (p[idx - 1] == 1) & (p[idx + 1] == 1) & (p[idx + 2] == 1)
            same = np.all([pn[idx + o] == (not wanted_air) for o in (-3, -2, -1, 0, 1, 2)], axis=0)
            around = np.median(np.stack([mag[idx - 2], mag[idx - 1], mag[idx + 1], mag[idx + 2]]), axis=0)
            usable = clean & same & (around >= 3.0)
            for lo, hi in buckets:
                chosen = usable & (p[idx] >= lo) & (p[idx] <= hi)
                if lo == 1:
                    chosen &= (idx % 25 == 0)
                ratios[(lo, hi)].extend((mag[idx][chosen] / around[chosen]).tolist())
        for b in buckets:
            count = int(((periods >= b[0]) & (periods <= b[1]) & (air == wanted_air) & ~reset).sum())
            table[(wanted_air, b)] = (count, ratios[b])
            
    for lo, hi in buckets:
        name = "continuo" if lo == 1 else (f"{lo} periodi" if lo == hi else f"{lo}-{hi if hi < 9999 else '...'} periodi")
        cells = []
        for wanted_air in (False, True):
            count, values = table[(wanted_air, (lo, hi))]
            ratio = f"{np.median(values):8.2f}" if len(values) >= 5 else f"{'-':>8s}"
            cells.append(f"{count if lo > 1 else '':>8} {ratio} {'(' + str(len(values)) + ')':>9s}")
        print(f"   {name:>16s} | {cells[0]} | {cells[1]}")
        
    surface_2 = table[(False, (2, 2))][1]
    if len(surface_2) >= 20:
        checks["I gap su superficie sono solo timestamp in ritardo (rapporto ~1)"] = float(np.median(surface_2)) < 1.4

    # [3] confronto tra le tre versioni degli offset
    print("\nPARTE 3: offset grezzi, divisi a ogni gap e con le regole finali")
    print(f"   {'':24s} {'scala':>7s} | {'SU SUPERFICIE':^21s} | {'IN ARIA':^21s} | {'oltre':>6s} {'oltre':>6s} | offset superficie")
    print(f"   {'versione':24s} {'(unità)':>7s} | {'99.9%':>10s} {'max':>10s} | {'99.9%':>10s} {'max':>10s} | {BIG:>6.0f} {LIMIT:>6.0f} | modificati")
    
    final_sizes = None
    for name, a, b in (("offset grezzi", "dx", "dy"), ("divisi a ogni gap", "ex", "ey"), ("regole finali", "fx", "fy")):
        scale = scale_of(rows, a, b)
        size = size_in_sigma(rows, a, b, scale)
        changed = int(sum(((r[a] != r["dx"]) | (r[b] != r["dy"]))[~r["air"]].sum() for r in rows))
        print(f"   {name:24s} {scale:7.2f} | {high(size[~air], 0.999):10.2f} {high(size[~air]):10.1f} | "
              f"{high(size[air], 0.999):10.2f} {high(size[air]):10.1f} | "
              f"{int((size > BIG).sum()):6d} {int((size > LIMIT).sum()):6d} | {changed}")
        if a == "fx":
            final_scale, final_sizes = scale, size
            
    checks[f"Nessun offset oltre {LIMIT:.0f} deviazioni standard"] = float(final_sizes.max()) <= LIMIT

    divisor, is_reset = cat(rows, "divisor"), cat(rows, "is_reset")
    n_changed = int(((divisor > 1) | is_reset).sum())
    print(f"\n   Regole finali: {n_changed:,} offset modificati ({100 * n_changed / n_all:.2f}% dei campioni, "
          f"tutti con la penna in aria)")
    
    for name, lo, hi in (("gap di 2-3 periodi", 2, 3), ("gap di 4+ periodi ", 4, 10 ** 18)):
        removed = []
        for r in rows:
            chosen = (r["divisor"] >= lo) & (r["divisor"] <= hi)
            removed.append(float((np.hypot(r["dx"], r["dy"]) * (1 - 1 / r["divisor"]))[chosen].sum()))
        count = int(((divisor >= lo) & (divisor <= hi)).sum())
        print(f"      {name}: {count:6,} offset;  tratto tolto alla traiettoria ricostruita: mediana {np.median(removed):7.0f} "
              f"unità, massimo {np.max(removed):7.0f}")
    print(f"      reset del clock, offset messi a zero: {int(is_reset.sum()):6,}")

    # [4] offset rimasti oltre BIG deviazioni standard
    print(f"\nPARTE 4: offset rimasti oltre {BIG:.0f} deviazioni standard, raggruppati in eventi")
    found = []
    for r in rows:
        fx, fy, pn, n = r["fx"] / final_scale, r["fy"] / final_scale, r["pen"], r["n"]
        start, stop = r["span"]
        for i in np.where(np.maximum(np.abs(fx), np.abs(fy)) > BIG)[0]:
            i = int(i)
            this = float(np.hypot(fx[i], fy[i]))
            nx, ny = (fx[i + 1], fy[i + 1]) if i + 1 < n else (0.0, 0.0)
            px, py = (fx[i - 1], fy[i - 1]) if i >= 1 else (0.0, 0.0)
            nxt, prv = float(np.hypot(nx, ny)), float(np.hypot(px, py))
            
            follows = lambda ox, oy, om: om >= 0.5 * this and (fx[i] * ox + fy[i] * oy) > 0.7 * this * om
            
            if i < start:
                kind = "prima del primo contatto penna"
            elif i >= stop:
                kind = "dopo l'ultimo contatto penna"
            elif np.hypot(fx[i] + nx, fy[i] + ny) < 0.3 * this or np.hypot(fx[i] + px, fy[i] + py) < 0.3 * this:
                kind = "picco (va e torna subito)"
            elif follows(nx, ny, nxt) or follows(px, py, prv):
                kind = "movimento veloce prolungato (sembra reale)"
            elif r["step"][i] <= 0:
                kind = "salto singolo dove il clock torna indietro"
            elif r["divisor"][i] > 1:
                kind = "salto singolo a un gap in aria (ancora grande dopo la divisione)"
            elif not r["air"][i]:
                kind = "salto singolo con la penna sulla superficie (mai modificato)"
            else:
                kind = "salto singolo in aria, senza gap"
                
            pens = f"{int(pn[i - 1]) if i else '-'}{int(pn[i])}{int(pn[i + 1]) if i + 1 < n else '-'}"
            found.append({"size": this, "kind": kind, "rec": (r["pid"], r["task"]), "i": i, "n": n,
                          "class": "PD" if r["label"] else "H ", "pens": pens, "step": float(r["step"][i]),
                          "dxdy": (float(fx[i]), float(fy[i])), "next": (float(nx), float(ny))})

    found.sort(key=lambda f: (f["rec"], f["kind"], f["i"]))
    events = []
    for f in found:
        last = events[-1] if events else None
        if last and last[-1]["rec"] == f["rec"] and last[-1]["kind"] == f["kind"] and f["i"] - last[-1]["i"] <= 3:
            last.append(f)
        else:
            events.append([f])
            
    events.sort(key=lambda e: -max(f["size"] for f in e))
    for e in events[:60]:
        top = max(e, key=lambda f: f["size"])
        where = f"campione {top['i']}/{top['n']}" if len(e) == 1 else (
            f"campioni {e[0]['i']} -> {e[-1]['i']} su {top['n']}  "
            f"({len(e)} offset, il più piccolo {min(f['size'] for f in e):.1f})")
        detail = (f"  penna [{top['pens']}]  passo {top['step']:8.3g}  dx,dy {top['dxdy'][0]:7.1f},{top['dxdy'][1]:7.1f}  "
                  f"successivo {top['next'][0]:7.1f},{top['next'][1]:7.1f}") if len(e) == 1 else ""
        print(f"   {top['size']:6.1f} dev. std  paziente {top['rec'][0]} task {top['rec'][1]}  {top['class']}  {where}{detail}  -> {top['kind']}")
        
    if len(events) > 60:
        print(f"   ... e altri {len(events) - 60} eventi")
    print(f"   {len(found)} offset raggruppati in {len(events)} eventi, in {len({f['rec'] for f in found})} registrazioni")
    
    for kind, count in Counter(f["kind"] for f in found).most_common():
        print(f"      {count:4d} offset in {sum(1 for e in events if e[0]['kind'] == kind):3d} eventi: {kind}")

    # [5] effetto di TRIM_AIR_EDGES
    print("\nPARTE 5: cosa toglierebbe TRIM_AIR_EDGES (tratti in aria prima del primo e dopo l'ultimo contatto)")
    for label, name in ((0, "Sani"), (1, "PD")):
        mine = [r for r in rows if r["label"] == label]
        lead = np.array([r["span"][0] for r in mine])
        trail = np.array([r["n"] - r["span"][1] for r in mine])
        total = sum(r["n"] for r in mine)
        print(f"   {name:8s}: {100 * (lead.sum() + trail.sum()) / total:4.1f}% dei campioni  "
              f"per registrazione -> inizio: mediana {np.median(lead):.0f} (max {lead.max()}) |  "
              f"fine: mediana {np.median(trail):.0f} (max {trail.max()})")
              
    inside = [f["size"] for f in found if "contatto penna" not in f["kind"]]
    print(f"   offset oltre {BIG:.0f} deviazioni standard che resterebbero dopo il taglio: {len(inside)} su {len(found)}"
          + (f", il più grande {max(inside):.1f}" if inside else ""))
          
    if dp.TRIM_AIR_EDGES:
        checks[f"Nessun offset oltre {LIMIT:.0f} deviazioni standard"] = max(inside, default=0.0) <= LIMIT

    # [6] dataset_prep.py applica le regole? confronto con l'export precedente
    print("\nPARTE 6: controllo di dataset_prep.py e confronto con l'export precedente")
    matches, surface_plain = True, True
    for r in rows:
        start, stop = r["span"] if dp.TRIM_AIR_EDGES else (0, r["n"])
        want = np.stack([r["fx"], r["fy"]], axis=1)[start:stop].astype(np.float32)
        plain = np.stack([r["dx"], r["dy"]], axis=1)[start:stop].astype(np.float32)
        got = r["produced"][:, :2]
        keep = np.ones(len(got), dtype=bool)
        keep[0] = not dp.TRIM_AIR_EDGES
        matches &= got.shape == want.shape and np.array_equal(got[keep], want[keep])
        on_surface = ~r["air"][start:stop] & keep
        surface_plain &= got.shape == plain.shape and np.array_equal(got[on_surface], plain[on_surface])
        
    print(f"   dataset_prep.py produce esattamente le regole descritte              : {'PASS' if matches else 'FAIL'}")
    print(f"   ogni offset su superficie è uguale alla differenza grezza, bit per bit: {'PASS' if surface_plain else 'FAIL'}")
    checks["dataset_prep.py applica esattamente le regole sui gap"] = matches and surface_plain

    old_file, map_file = Path("packed_export/pahaw_packed.npz"), Path("Cache/id_map.json")
    if not (old_file.exists() and map_file.exists()):
        print("   confronto saltato: in locale non c'è un export precedente")
    else:
        with open(map_file) as f:
            original_id = json.load(f)["new_to_original"]
        old = {(original_id[pid], task + 1): features for pid, task, features in read_old_export(old_file)}
        same_set = old.keys() == {(r["pid"], r["task"]) for r in rows}
        other_same = same_set
        made_with = Counter()
        for r in rows:
            if not same_set:
                break
            before = old[(r["pid"], r["task"])]
            now = r["produced"]
            if len(before) != len(now):
                start, stop = r["span"]
                before = before[start:stop] if len(before) > len(now) else before
                now = now if len(before) == len(now) else now[start:stop]
            other_same &= before.shape == now.shape and np.array_equal(before[1:, 2:], now[1:, 2:])
            if before.shape[0] == r["n"]:
                for name, a, b in (("RAW", "dx", "dy"), ("Divisione", "ex", "ey"), ("Regole Finali", "fx", "fy")):
                    if np.array_equal(before[:, :2], np.stack([r[a], r[b]], axis=1).astype(np.float32)):
                        made_with[name] += 1
                        
        print(f"   stessi pazienti e task: {'PASS' if same_set else 'FAIL'};  "
              f"canali diversi da dx, dy invariati: {'PASS' if other_same else 'FAIL'}")
        print(f"   regole con cui era stato fatto l'export precedente: {dict(made_with) or 'non riconosciute'} (registrazioni)")
        checks["Rispetto all'export precedente cambiano solo dx e dy"] = same_set and other_same
        if not (same_set and other_same):
            print("   -> ATTENZIONE: forse è cambiata un'impostazione di dataset_prep.py (LABELS_CSV, SVC_COLUMNS, RAW_DATA_DIR).")

    print("\nVERDETTO")
    for name, passed in checks.items():
        print(f"   [{'PASS' if passed else 'FAIL'}] {name}")
    if all(checks.values()):
        print("   Tutti i controlli superati. Si può esportare con python export_packed_data.py.")
    else:
        print("   Alcuni controlli NON superati: non addestrare su questi dati prima di aver capito perché.")


if __name__ == "__main__":
    main()