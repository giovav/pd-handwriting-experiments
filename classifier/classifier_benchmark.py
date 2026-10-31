"""
classifier_benchmark.py: valutazione dei dataset sintetici con il classificatore di
classifier_adapter.py (ingresso "kinematic").

Comandi:
    run       (train on real, test on synthetic) per ogni fold addestra il classificatore
              sui pazienti di training reali e lo valuta sui pazienti di test reali, sui
              pazienti di training reali e su ogni dataset sintetico del fold;
    tstr      (train on synthetic, test on real) per ogni dataset sintetico addestra un
              classificatore su quei dati (validazione sui pazienti reali di validazione)
              e lo valuta sui pazienti di test reali, accanto a quello addestrato sui reali;
    aggregate media e deviazione standard sui fold, tabelle e grafici.
Metriche a livello di registrazione, per task e, per i dati reali, per paziente (media
delle probabilità delle sue registrazioni). Prima del training check_inputs controlla ID
e fold dei file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# I moduli condivisi stanno in common/ (se c'è una copia accanto allo script, vale quella).
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "common"))

import classifier_adapter as adapter
from create_split import load_fold
from packed_data import load_packed

METRICS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]


def find_synthetic(synthetic_dir: str | None, files: list[str]) -> list[tuple[str, int, Path]]:
    """
    Trova i dataset sintetici <modello>/fold<k>[tag].npz.

    Restituisce (nome del dataset, fold, percorso); il nome è la cartella del modello più
    l'eventuale tag (es. 'diffink_cfg1').
    """
    paths = [Path(f) for f in files]
    if synthetic_dir:
        paths += sorted(p for p in Path(synthetic_dir).glob("*/fold*.npz") if ".tmp" not in p.name)
    found = []
    for p in paths:
        match = re.fullmatch(r"fold(\d+)(.*)", p.stem)
        if not match:
            raise SystemExit(f"{p}: il nome del file deve essere fold<k>[tag].npz")
        found.append((p.parent.name + match.group(2), int(match.group(1)), p))
    return found


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC dai ranghi; i valori pari ricevono il rango medio."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    values = np.concatenate([pos, neg])[order]
    ranks[order] = np.arange(1, len(order) + 1)
    
    for v in np.unique(values):
        tied = np.concatenate([pos, neg]) == v
        ranks[tied] = ranks[tied].mean()
        
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Metriche alla soglia data: accuracy, balanced accuracy, sensitivity, specificity,
    precision, F1 e AUC (NaN quando non sono definite).
    """
    pred = (p >= threshold).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    ratio = lambda a, b: a / b if b else float("nan")
    
    sens, spec, prec = ratio(tp, tp + fn), ratio(tn, tn + fp), ratio(tp, tp + fp)
    
    return {"n": len(y), "n_PD": int(y.sum()), "accuracy": ratio(tp + tn, len(y)),
            "balanced_accuracy": float(np.nanmean([sens, spec])) if np.isfinite([sens, spec]).any() else float("nan"),
            "sensitivity": sens, "specificity": spec, "precision": prec,
            "f1": ratio(2 * prec * sens, prec + sens) if np.isfinite([prec, sens]).all() else float("nan"),
            "auc": roc_auc(y, p), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def evaluate(model_dir: Path, recordings: list[dict], is_real: bool, device: str) -> tuple[dict, list[dict]]:
    """
    Probabilità per registrazione dal classificatore e metriche per registrazione, per task
    e, se patient_level, per paziente.
    """
    p = np.asarray(adapter.predict(model_dir, recordings, device), dtype=float)
    assert p.shape == (len(recordings),) and np.isfinite(p).all(), "predict deve restituire una probabilità per registrazione"
    
    y = np.array([r["label"] for r in recordings])
    tasks = np.array([r["task"] for r in recordings])
    model_epoch = json.loads((Path(model_dir) / "info.json").read_text())["eval_epoch"]
    
    result = {"model_epoch": model_epoch, "recording": classification_metrics(y, p),
              "per_task": {str(t + 1): classification_metrics(y[tasks == t], p[tasks == t]) for t in np.unique(tasks)}}
              
    if is_real:
        by_patient = defaultdict(list)
        for r, prob in zip(recordings, p):
            by_patient[(r["patient_id"], r["label"])].append(prob)
        keys = sorted(by_patient)
        result["patient"] = classification_metrics(np.array([k[1] for k in keys]),
                                                   np.array([np.mean(by_patient[k]) for k in keys]))
                                                   
    predictions = [{"patient_id": r["patient_id"], "task": r["task"] + 1, "label": r["label"], "p_pd": float(prob)}
                   for r, prob in zip(recordings, p)]
                   
    return result, predictions


def ensure_trained(model_dir: Path, train: list[dict], val: list[dict], args, what: str) -> None:
    """
    Addestra il classificatore, a meno che non esista già (file DONE) e non sia dato --overwrite.
    """
    if (model_dir / "DONE").exists() and not args.overwrite:
        print(f"   {what}: già addestrato ({model_dir}), lo riuso")
        return
        
    print(f"   {what}: training su {len(train)} registrazioni (validazione {len(val)}) -> {model_dir}")
    start = time.time()
    
    adapter.train(train, val, model_dir, seed=args.seed, smoke=args.smoke, device=args.device)
    (model_dir / "DONE").write_text(json.dumps({"classifier": adapter.NAME, "seconds": round(time.time() - start)}))


def save_result(out_dir: Path, mode: str, fold: int, trained_on: str, evaluated: str, result: dict,
                predictions: list[dict]) -> None:
    """Salva il risultato in JSON e le previsioni per registrazione in CSV."""
    folder = out_dir / "results" / mode / f"fold{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{trained_on}__on__{evaluated}".replace(" ", "_")
    
    (folder / f"{stem}.json").write_text(json.dumps({"mode": mode, "fold": fold, "trained_on": trained_on,
                                                     "evaluated": evaluated, "classifier": adapter.NAME,
                                                     "input_set": adapter.INPUT_SET, **result},
                                                    indent=1))
                                                    
    with open(folder / f"{stem}_predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
        
    m = result["recording"]
    print(f"      {trained_on:>22s} -> {evaluated:22s} n={m['n']:4d}  Bal.Acc {m['balanced_accuracy']:.3f}  "
          f"Sens {m['sensitivity']:.3f}  Spec {m['specificity']:.3f}  AUC {m['auc']:.3f}")


PSEUDONYM = re.compile(r"P\d{3}")

def check_inputs(real: list[dict], split_file: str, folds: list[int], synthetic: list) -> None:
    """
    Si ferma prima del training se:
        - il file packed o lo split contengono ID che non sono pseudonimi P + 3 cifre;
        - un paziente dello split non è nel file packed;
        - una registrazione sintetica non ha un ID SYN-, o il .json accanto a un file
          sintetico indica un fold diverso da quello nel nome del file.
    """
    problems = []
    real_ids = {r["patient_id"] for r in real}
    bad = sorted(i for i in real_ids if not PSEUDONYM.fullmatch(i))
    if bad:
        problems.append(f"il file packed contiene ID che non sono pseudonimi (P + 3 cifre): {bad[:5]}")
        
    for fold in folds:
        split = load_fold(fold, split_file)
        listed = set(split["train"]) | set(split["val"]) | set(split["test"])
        
        if not_pseudo := sorted(i for i in listed if not PSEUDONYM.fullmatch(i)):
            problems.append(f"Fold {fold}: lo split contiene ID che non sono pseudonimi: {not_pseudo[:5]}")
            
        if missing := sorted(listed - real_ids):
            problems.append(f"Fold {fold}: pazienti dello split assenti dal file packed: {missing[:5]} "
                            f"(split e file packed devono venire dallo stesso export)")
                            
    for name, fold, path in synthetic:
        if fold not in folds:
            continue
            
        ids = {r["patient_id"] for r in load_packed(path)}
        if not all(i.startswith("SYN-") for i in ids):
            problems.append(f"{path}: registrazioni sintetiche senza ID SYN-: {sorted(ids)[:3]}")
            
        side = path.with_suffix(".json")
        if side.exists():
            meta_fold = json.loads(side.read_text()).get("fold")
            if meta_fold is not None and int(meta_fold) != fold:
                problems.append(f"{path}: il nome del file indica il fold {fold}, il .json il fold {meta_fold}")
        else:
            print(f"   Nota: {side.name} non trovato, il fold è preso dal nome del file.")
            
    if problems:
        raise SystemExit("Problemi nei file di ingresso:\n   " + "\n   ".join(problems))
        
    print(f"Ingressi controllati: {len(real_ids)} pazienti pseudonimizzati, fold {folds}, "
          f"{sum(f in folds for _, f, _ in synthetic)} file sintetici con ID SYN-")


def prepare(args) -> tuple[Path, list[dict], list]:
    """Controlli e caricamento comuni a run e tstr."""
    out_dir = Path(args.out_dir)
    real = load_packed(args.data_file)
    synthetic = find_synthetic(args.synthetic_dir, [])
    
    if not synthetic:
        raise SystemExit(f"Nessun file sintetico in {args.synthetic_dir}/<modello>/fold<k>*.npz")
        
    check_inputs(real, args.split_file, args.folds, synthetic)
    print(f"Classificatore: {adapter.NAME}\nDataset sintetici: {sorted({n for n, _, _ in synthetic})}")
    return out_dir, real, synthetic


def fold_parts(real: list[dict], fold: int, split_file: str) -> dict:
    """Registrazioni reali di train, val e test del fold."""
    split = load_fold(fold, split_file)
    assert not set(split["test"]) & (set(split["train"]) | set(split["val"])), "un paziente di test è anche in train o val"
    return {name: [r for r in real if r["patient_id"] in set(split[name])] for name in ("train", "val", "test")}


def run(args) -> None:
    """
    run: classificatore addestrato sui reali, valutato su test reale, train reale e dati sintetici.
    """
    out_dir, real, synthetic = prepare(args)
    for fold in args.folds:
        part = fold_parts(real, fold, args.split_file)
        here = [(name, path) for name, f, path in synthetic if f == fold]
        
        print(f"\nFold {fold} [run]: {len(part['train'])} / {len(part['val'])} / {len(part['test'])} registrazioni reali "
              f"(train / val / test); dataset sintetici: {[n for n, _ in here] or 'nessuno'}")
              
        real_model = out_dir / "models" / "real" / f"fold{fold}"
        ensure_trained(real_model, part["train"], part["val"], args, "classificatore sui dati reali")
        
        for evaluated, recs in (("real test", part["test"]), ("real train", part["train"])):
            save_result(out_dir, "trts", fold, "real", evaluated, *evaluate(real_model, recs, True, args.device))
            
        for name, path in here:
            save_result(out_dir, "trts", fold, "real", name, *evaluate(real_model, load_packed(path), False, args.device))


def tstr(args) -> None:
    """tstr: un classificatore per ogni dataset sintetico, valutato sui pazienti di test reali."""
    out_dir, real, synthetic = prepare(args)
    for fold in args.folds:
        part = fold_parts(real, fold, args.split_file)
        here = [(name, path) for name, f, path in synthetic if f == fold]
        
        print(f"\nFold {fold} [tstr]: {len(part['test'])} registrazioni reali di test; dataset sintetici: {[n for n, _ in here] or 'nessuno'}")
        real_model = out_dir / "models" / "real" / f"fold{fold}"
        ensure_trained(real_model, part["train"], part["val"], args, "classificatore sui dati reali (riferimento)")
        
        save_result(out_dir, "tstr", fold, "real", "real test", *evaluate(real_model, part["test"], True, args.device))
        for name, path in here:
            model_dir = out_dir / "models" / "synthetic" / name / f"fold{fold}"
            ensure_trained(model_dir, load_packed(path), part["val"], args, f"classificatore su {name}")
            save_result(out_dir, "tstr", fold, name, "real test", *evaluate(model_dir, part["test"], True, args.device))


def aggregate(args) -> None:
    """
    Media e deviazione standard sui fold di ogni risultato; summary.csv, summary.md, summary.png.
    """
    out_dir = Path(args.out_dir)
    files = sorted((out_dir / "results").glob("*/fold*/*.json"))
    if not files:
        raise SystemExit(f"Nessun risultato in {out_dir}/results: eseguire prima run o tstr.")
        
    groups = defaultdict(list)
    for f in files:
        r = json.loads(f.read_text())
        groups[(r["mode"], r["trained_on"], r["evaluated"])].append(r)

    rows = []
    for (mode, trained_on, evaluated), results in sorted(groups.items()):
        classifiers = {r["classifier"] for r in results}
        for level in ("recording", "patient"):
            if level not in results[0]:
                continue
            row = {"mode": mode, "trained_on": trained_on, "evaluated": evaluated, "level": level,
                   "folds": " ".join(str(r["fold"]) for r in sorted(results, key=lambda r: r["fold"])),
                   "n_per_fold": round(float(np.mean([r[level]["n"] for r in results])), 1),
                   "classifier": " | ".join(sorted(classifiers))}
                   
            for m in METRICS:
                values = np.array([r[level][m] for r in results], dtype=float)
                row[f"{m}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
                row[f"{m}_sd"] = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else float("nan")
            rows.append(row)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        
    lines = []
    for mode, title in (("trts", "TRAIN ON REAL, TEST ON SYNTHETIC"),
                        ("tstr", "TRAIN ON SYNTHETIC, TEST ON REAL (pazienti di test)")):
        sel = [r for r in rows if r["mode"] == mode]
        if not sel:
            continue
        lines += [f"## {title}", "", "| trained on | evaluated on | level | folds | n/fold | "
                  + " | ".join(m.replace("_", " ") for m in METRICS) + " |",
                  "|" + "---|" * (5 + len(METRICS))]
        for r in sel:
            cells = [f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_sd']:.3f}" if np.isfinite(r[f"{m}_sd"])
                     else f"{r[f'{m}_mean']:.3f}" for m in METRICS]
            lines.append(f"| {r['trained_on']} | {r['evaluated']} | {r['level']} | {r['folds']} | {r['n_per_fold']} | "
                         + " | ".join(cells) + " |")
        lines.append("")
        
    lines.append(f"Classificatore: {' / '.join(sorted({r['classifier'] for r in rows}))}. Media ± SD sui fold, soglia 0.5.")
    (out_dir / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    
    plot_summary(rows, out_dir / "summary.png")
    saved = ["summary.csv", "summary.md", "summary.png"]
    if any(mode == "trts" for mode, _, _ in groups):
        saved += trts_by_model(groups, out_dir)
    print(f"\nFatto. File in {out_dir}: {', '.join(saved)}")


REAL_SETS = ("real test", "real train")
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]


def trts_by_model(groups: dict, out_dir: Path) -> list[str]:
    """
    run: per ogni modello, confronto con test e train reali degli stessi fold e differenza per fold
    (trts_by_model.csv/.png, trts_per_task.csv/.png).
    """
    by_fold = {evaluated: {r["fold"]: r for r in results}
               for (mode, trained_on, evaluated), results in groups.items() if mode == "trts" and trained_on == "real"}
    models = sorted(e for e in by_fold if e not in REAL_SETS)
    if not models or "real test" not in by_fold:
        return []
        
    rows, task_rows = [], []
    for model in models:
        folds = sorted(f for f in by_fold[model] if f in by_fold["real test"])
        missing = sorted(set(by_fold["real test"]) - set(folds))
        for m in METRICS:
            v = {s: np.array([by_fold[s][f]["recording"][m] if f in by_fold.get(s, {}) else np.nan for f in folds])
                 for s in (model, *REAL_SETS)}
            d = v[model] - v["real test"]
            
            rows.append({"model": model, "metric": m, "folds": " ".join(map(str, folds)),
                         "folds_missing": " ".join(map(str, missing)),
                         "synthetic_mean": _mean(v[model]), "synthetic_sd": _sd(v[model]),
                         "real_test_mean": _mean(v["real test"]), "real_test_sd": _sd(v["real test"]),
                         "real_train_mean": _mean(v["real train"]), "real_train_sd": _sd(v["real train"]),
                         "synthetic_minus_real_test_mean": _mean(d), "synthetic_minus_real_test_sd": _sd(d),
                         "per_fold_synthetic": " ".join(f"{x:.3f}" for x in v[model])})
                         
        for task in range(1, 9):
            for s in (model, "real test"):
                vals = [by_fold[s][f]["per_task"].get(str(task)) for f in folds]
                vals = [x for x in vals if x]
                if vals:
                    task_rows.append({"model": model, "evaluated": s, "task": task, "folds": " ".join(map(str, folds)),
                                      **{f"{m}_mean": _mean(np.array([x[m] for x in vals])) for m in METRICS}})
                                      
    _write_rows(out_dir / "trts_by_model.csv", rows)
    _write_rows(out_dir / "trts_per_task.csv", task_rows)
    plot_by_model(rows, by_fold, models, out_dir / "trts_by_model.png")
    plot_per_task(task_rows, models, out_dir / "trts_per_task.png")
    
    for r in rows:
        if r["folds_missing"] and r["metric"] == METRICS[0]:
            print(f"Nota: {r['model']} non ha dati sintetici per i fold {r['folds_missing']}; "
                  f"il confronto usa solo i fold {r['folds']}")
                  
    return ["trts_by_model.csv", "trts_per_task.csv", "trts_by_model.png", "trts_per_task.png"]


def _mean(v: np.ndarray) -> float:
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def _sd(v: np.ndarray) -> float:
    return float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float("nan")


def _write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows])


def plot_by_model(rows: list[dict], by_fold: dict, models: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    bars = [("real test", "real_test", "#3b3b3b"), ("real train", "real_train", "#b0aea8")]
    
    for ax, metric in zip(axes.flat, METRICS):
        for i, model in enumerate(models):
            r = next(x for x in rows if x["model"] == model and x["metric"] == metric)
            folds = [int(f) for f in r["folds"].split()]
            series = bars + [(model, "synthetic", MODEL_COLORS[i % len(MODEL_COLORS)])]
            for k, (source, key, color) in enumerate(series):
                x = i + (k - 1) * 0.27
                mean, sd = r[f"{key}_mean"], r[f"{key}_sd"]
                ax.bar(x, mean, 0.25, color=color, yerr=0 if not np.isfinite(sd) else sd,
                       error_kw={"lw": 1, "ecolor": "#52514e"}, label=source if k < 2 and i == 0 else None)
                dots = [by_fold[source][f]["recording"][metric] for f in folds if f in by_fold.get(source, {})]
                ax.plot([x] * len(dots), dots, "o", ms=3, color="white", mec="#1a1a19", mew=0.6)
                
        ax.axhline(0.5, color="#9a9a96", lw=1, ls="--")
        ax.set_xticks(range(len(models)), [f"{m}\n(folds {next(x['folds'] for x in rows if x['model'] == m)})"
                                           for m in models], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric.replace("_", " "), fontsize=10)
        ax.tick_params(labelsize=7)
        
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[0]))
    labels.append("dataset sintetici (un colore per modello)")
    axes.flat[-1].legend(handles, labels, loc="center", fontsize=9, frameon=False)
    fig.suptitle("Classificatore addestrato sul train reale, valutato su ogni dataset sintetico "
                 "(livello registrazione, soglia 0.5; i punti sono i singoli fold)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_per_task(task_rows: list[dict], models: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("balanced_accuracy", "-", "o"), ("sensitivity", "--", "s"), ("specificity", ":", "^")]
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.8), squeeze=False)
    
    for i, (ax, model) in enumerate(zip(axes[0], models)):
        for source, color in (("real test", "#3b3b3b"), (model, MODEL_COLORS[i % len(MODEL_COLORS)])):
            sel = sorted((r for r in task_rows if r["model"] == model and r["evaluated"] == source), key=lambda r: r["task"])
            for metric, ls, marker in metrics:
                ax.plot([r["task"] for r in sel], [r[f"{metric}_mean"] for r in sel], ls=ls, marker=marker, ms=3,
                        color=color, lw=1, label=f"{'real test' if source == 'real test' else 'synthetic'}: "
                                                 f"{metric.replace('_', ' ')}")
                                                 
        ax.axhline(0.5, color="#9a9a96", lw=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(1, 9))
        ax.set_xlabel("task", fontsize=8)
        ax.set_title(model, fontsize=10)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, frameon=False, loc="lower left")
        
    fig.suptitle("Risultati per task, media sui fold (sensitivity = quota di registrazioni sintetiche PD riconosciute come PD)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_summary(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [(mode, [r for r in rows if r["mode"] == mode and r["level"] == "recording"]) for mode in ("trts", "tstr")]
    panels = [(m, sel) for m, sel in panels if sel]
    metrics = [("balanced_accuracy", "#2a78d6"), ("sensitivity", "#eb6834"), ("specificity", "#1baf7a")]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.2), squeeze=False)
    
    for ax, (mode, sel) in zip(axes[0], panels):
        names = [r["evaluated"] if mode == "trts" else f"Trained-on {r['trained_on']}" for r in sel]
        x = np.arange(len(sel))
        for k, (m, color) in enumerate(metrics):
            ax.bar(x + (k - 1) * 0.26, [r[f"{m}_mean"] for r in sel], 0.24, color=color, label=m.replace("_", " "),
                   yerr=[0 if not np.isfinite(r[f"{m}_sd"]) else r[f"{m}_sd"] for r in sel],
                   error_kw={"lw": 1, "ecolor": "#52514e"})
                   
        ax.axhline(0.5, color="#9a9a96", lw=1, ls="--")
        ax.set_xticks(x, names, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("media sui fold (± SD)", fontsize=8)
        ax.set_title("addestrato sui reali [run]" if mode == "trts" else "addestrato sui sintetici [tstr] (test reale)", fontsize=10)
        ax.tick_params(labelsize=8)
        
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark del classificatore sui dataset sintetici PaHaW.")
    sub = parser.add_subparsers(dest="command", required=True)
    
    for name, text in (("run", "train on real, test on synthetic"),
                       ("tstr", "train on synthetic, test on real")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--folds", type=int, nargs="+", required=True)
        p.add_argument("--data_file", type=str, required=True, help="file packed (es. pahaw_packed.npz)")
        p.add_argument("--split_file", type=str, required=True, help="split esportato insieme al file packed (es. splits_4fold.json)")
        p.add_argument("--synthetic_dir", type=str, required=True, help="cartella con <modello>/fold<k>*.npz")
        p.add_argument("--out_dir", type=str, default="classifier_results")
        p.add_argument("--device", type=str, default="cuda", help="device di PyTorch")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--smoke", action="store_true", help="prova rapida: poche epoche e poche registrazioni")
        p.add_argument("--overwrite", action="store_true", help="riaddestra anche i classificatori già completati")
        
    a = sub.add_parser("aggregate", help="riepilogo dei risultati: CSV, Markdown e grafici")
    a.add_argument("--out_dir", type=str, default="classifier_results")
    
    args = parser.parse_args()
    {"run": run, "tstr": tstr, "aggregate": aggregate}[args.command](args)


if __name__ == "__main__":
    main()