"""
classifier_adapter.py: il classificatore usato nel benchmark, dietro due funzioni,
train(...) e predict(...), che restituisce una probabilità di Parkinson per registrazione.

Modello: CNN_BiLSTMModel di baseline_models.py, su finestre di 256 campioni. Ingresso
"kinematic" = [vx, vy, speed, pressure, dp, pen], fissato a priori e
calcolato allo stesso modo per dati reali e sintetici: vx = dx, vy = dy,
speed = sqrt(vx^2 + vy^2), dp = variazione di pressione tra due campioni consecutivi con la
penna appoggiata (0 altrimenti).

Normalizzazione per task, calcolata solo sulle registrazioni di training: vx, vy media 0 e
una scala comune (RMS); speed e dp media 0 e la propria RMS (dp sui soli campioni con la
penna appoggiata); pressure media e deviazione standard sui campioni con la penna
appoggiata; pen resta 0/1; valori limitati a +-8 deviazioni standard.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

import dataset_prep as dp
from baseline_models import CNN_BiLSTMModel

CHANNELS = ["vx", "vy", "speed", "pressure", "dp", "pen"]
INPUT_SET = "kinematic"
NAME = (f"Baseline CNN_BiLSTM - Input: '{INPUT_SET}' {CHANNELS} (256-win, per-task norm su train, "
        f"crop & rotazioni casuali, AdamW, salvataggio a epoch 7)")

WINDOW_SIZE = 256
STEP_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 35
EVAL_EPOCH = 7                       
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2                  
GRAD_CLIP = 1.0
CLIP_SD = 8.0
ROTATE_DEG = 10.0
SMOKE_EPOCHS, SMOKE_RECORDINGS = 2, 48
PREDICT_BATCH = 256

VX, VY, SPEED, PRESSURE, DP, PEN = range(len(CHANNELS))
PEN_RAW = dp.CHANNEL_NAMES.index("pen")


def representation(features: np.ndarray) -> np.ndarray:
    """Registrazione [T, 6] -> canali [vx, vy, speed, pressure, dp, pen] (float64)."""
    assert features.ndim == 2 and features.shape[1] == dp.N_CHANNELS, f"attesa una registrazione [T, 6], trovata {features.shape}"
    assert dp.CHANNEL_NAMES == ["dx", "dy", "pressure", "azimuth", "altitude", "pen"]
    
    dx, dy = features[:, 0].astype(np.float64), features[:, 1].astype(np.float64)
    pressure = features[:, 2].astype(np.float64)
    pen = (features[:, 5] > 0.5).astype(np.float64)
    
    both_down = np.concatenate([[False], (pen[1:] > 0) & (pen[:-1] > 0)])
    d_pressure = np.where(both_down, np.diff(pressure, prepend=pressure[:1]), 0.0)
    
    return np.column_stack([dx, dy, np.hypot(dx, dy), pressure, d_pressure, pen])


def padded(x: np.ndarray) -> np.ndarray:
    """Registrazioni più corte di una finestra: si ripete l'ultimo campione fino a 256."""
    return x if len(x) >= WINDOW_SIZE else np.concatenate([x, np.repeat(x[-1:], WINDOW_SIZE - len(x), axis=0)])


def grid_windows(x: np.ndarray) -> np.ndarray:
    """
    Finestre [N, 256, C] con passo 128, più una finestra allineata alla fine della registrazione.
    """
    x = padded(x)
    starts = list(range(0, len(x) - WINDOW_SIZE + 1, STEP_SIZE))
    if starts[-1] != len(x) - WINDOW_SIZE:
        starts.append(len(x) - WINDOW_SIZE)
    return np.stack([x[s:s + WINDOW_SIZE] for s in starts])


def all_grid_windows(recordings: list[dict]):
    """Finestre di tutte le registrazioni, con l'indice della registrazione di origine e il task."""
    parts = [grid_windows(representation(r["features"])) for r in recordings]
    owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(parts)])
    tasks = np.array([r["task"] for r in recordings])[owner]
    return np.concatenate(parts), owner, tasks


def _stats(recordings: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Medie e scale dei canali (vedi la docstring del modulo)."""
    flat = np.concatenate([representation(r["features"]) for r in recordings])
    down = np.concatenate([r["features"][:, PEN_RAW] > 0.5 for r in recordings])
    
    mean, std = np.zeros(len(CHANNELS)), np.ones(len(CHANNELS))
    std[VX] = std[VY] = np.sqrt((np.mean(flat[:, VX] ** 2) + np.mean(flat[:, VY] ** 2)) / 2)
    std[SPEED] = np.sqrt(np.mean(flat[:, SPEED] ** 2))
    
    if down.any():
        mean[PRESSURE], std[PRESSURE] = flat[down, PRESSURE].mean(), flat[down, PRESSURE].std()
        std[DP] = np.sqrt(np.mean(flat[down, DP] ** 2))
        
    return mean, np.where(std < 1e-8, 1.0, std)


def fit_normalization(train_recordings: list[dict]) -> dict:
    """
    Statistiche per task sulle registrazioni di training, più quelle globali per un task mancante.
    """
    g_mean, g_std = _stats(train_recordings)
    per_task = {}
    
    for t in range(dp.N_TASKS):
        recs = [r for r in train_recordings if r["task"] == t]
        if recs:
            mean, std = _stats(recs)
            per_task[str(t)] = {"mean": mean.tolist(), "std": std.tolist()}
            
    return {"per_task": per_task, "global": {"mean": g_mean.tolist(), "std": g_std.tolist()}, "clip_sd": CLIP_SD}


def normalize(w: np.ndarray, tasks: np.ndarray, norm: dict) -> np.ndarray:
    """Normalizza le finestre con le statistiche del loro task e limita i valori a +-clip_sd."""
    out = np.empty(w.shape, dtype=np.float32)
    for t in np.unique(tasks):
        s = norm["per_task"].get(str(int(t)), norm["global"])
        sel = tasks == t
        out[sel] = (w[sel] - np.array(s["mean"])) / np.array(s["std"])
    return np.clip(out, -norm["clip_sd"], norm["clip_sd"])


def to_tensor(w: np.ndarray) -> torch.Tensor:
    """[N, 256, C] -> tensore [N, C, 256] per Conv1d."""
    return torch.from_numpy(np.ascontiguousarray(w.transpose(0, 2, 1)))


def build_model() -> nn.Module:
    """CNN_BiLSTMModel con i canali dell'ingresso kinematic."""
    return CNN_BiLSTMModel(input_channels=len(CHANNELS))


_WARNED_NO_CUDA = False

def resolve_device(device: str) -> torch.device:
    """Se CUDA non c'è, usa la CPU (con un avviso una volta sola)."""
    global _WARNED_NO_CUDA
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        if not _WARNED_NO_CUDA:
            print("      (CUDA non disponibile: uso la CPU)")
            _WARNED_NO_CUDA = True
        return torch.device("cpu")
    return torch.device(device)


@torch.no_grad()
def window_probabilities(model: nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    """Probabilità per finestra, calcolate a blocchi di PREDICT_BATCH."""
    model.eval()
    out = [torch.sigmoid(model(x[i:i + PREDICT_BATCH].to(device))).cpu() for i in range(0, len(x), PREDICT_BATCH)]
    return torch.cat(out).numpy().astype(np.float64)


def recording_probabilities(p_windows: np.ndarray, owner: np.ndarray, n: int) -> np.ndarray:
    """Probabilità per registrazione: media delle probabilità delle sue finestre."""
    return np.bincount(owner, weights=p_windows, minlength=n) / np.bincount(owner, minlength=n)


def train(train_recordings: list[dict], val_recordings: list[dict], model_dir: Path, seed: int,
          smoke: bool, device: str) -> None:
    """
    Addestra il classificatore.

    Normalizzazione calcolata solo sulle registrazioni di training. A ogni passo le finestre
    partono da una posizione casuale di una registrazione estratta con peso 1 / (registrazioni
    della sua classe), e vengono ruotate di un angolo casuale entro +-10 gradi. Si salva il
    modello dell'epoca EVAL_EPOCH; si addestra comunque per tutte le epoche per avere le
    curve di apprendimento complete.
    """
    from classifier_benchmark import classification_metrics

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = resolve_device(device)
    
    epochs = SMOKE_EPOCHS if smoke else EPOCHS
    eval_epoch = min(EVAL_EPOCH, epochs)
    
    if smoke:
        keep = rng.permutation(len(train_recordings))[:SMOKE_RECORDINGS]
        train_recordings = [train_recordings[i] for i in sorted(keep)]

    reps = [padded(representation(r["features"])) for r in train_recordings]
    norm = fit_normalization(train_recordings)
    n_grid = sum(len(grid_windows(x)) for x in reps)
    
    w_val, owner_val, t_val = all_grid_windows(val_recordings)
    x_val = to_tensor(normalize(w_val, t_val, norm))
    
    y_rec_train = np.array([r["label"] for r in train_recordings])
    y_rec_val = np.array([r["label"] for r in val_recordings])
    y_val = torch.from_numpy(y_rec_val[owner_val].astype(np.float64))
    task_train = np.array([r["task"] for r in train_recordings])
    
    assert set(np.unique(y_rec_train)) == {0, 1}, "il training richiede registrazioni di entrambe le classi"

    p_recording = 1.0 / np.bincount(y_rec_train)[y_rec_train]
    p_recording /= p_recording.sum()
    steps = n_grid // BATCH_SIZE

    def batch():
        """
        Un batch di training: finestre in posizione casuale e ruotate di un angolo casuale (vx, vy insieme).
        """
        which = rng.choice(len(reps), size=BATCH_SIZE, p=p_recording)
        w = np.empty((BATCH_SIZE, WINDOW_SIZE, len(CHANNELS)))
        
        for k, i in enumerate(which):
            s = rng.integers(0, len(reps[i]) - WINDOW_SIZE + 1)
            w[k] = reps[i][s:s + WINDOW_SIZE]
            
        angle = np.deg2rad(rng.uniform(-ROTATE_DEG, ROTATE_DEG, size=BATCH_SIZE))[:, None]
        vx = w[:, :, VX].copy()
        w[:, :, VX] = np.cos(angle) * vx - np.sin(angle) * w[:, :, VY]
        w[:, :, VY] = np.sin(angle) * vx + np.cos(angle) * w[:, :, VY]
        
        return (to_tensor(normalize(w, task_train[which], norm)),
                torch.from_numpy(y_rec_train[which].astype(np.float32)))

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"      input '{INPUT_SET}' {CHANNELS}; {len(train_recordings)} registrazioni di training ({n_grid} finestre), "
          f"{len(val_recordings)} di validazione ({len(x_val)} finestre); {epochs} epoche su {device}, "
          f"modello salvato all'epoca {eval_epoch}")

    history, start = [], time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        
        for _ in range(steps):
            x, y = batch()
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += loss.item()
            
        if epoch == eval_epoch:
            torch.save(model.state_dict(), model_dir / "model.pt")

        p_val_windows = window_probabilities(model, x_val, device)
        val_loss = float(nn.functional.binary_cross_entropy(torch.from_numpy(p_val_windows).clamp(1e-7, 1 - 1e-7),
                                                            y_val))
        m = classification_metrics(y_rec_val, recording_probabilities(p_val_windows, owner_val, len(val_recordings)))
        
        history.append({"epoch": epoch, "train_loss": total / max(steps, 1), "val_loss": val_loss,
                        **{f"val_{k}": m[k] for k in ("accuracy", "balanced_accuracy", "sensitivity", "specificity",
                                                      "precision", "f1", "auc")}})
                                                      
        print(f"      epoca {epoch:2d}/{epochs}  TrainLoss {history[-1]['train_loss']:.4f}  ValLoss {val_loss:.4f}  "
              f"Val AUC {m['auc']:.3f}  Val Bal.Acc {m['balanced_accuracy']:.3f}"
              f"{'  <- salvato' if epoch == eval_epoch else ''}  ({time.time() - start:.0f} s)")

    info = {"name": NAME, "architecture": "CNN_BiLSTMModel", "input_set": INPUT_SET, "input_channels": CHANNELS,
            "window_size": WINDOW_SIZE, "step_size": STEP_SIZE, "normalization": norm,
            "training": {"optimizer": "AdamW", "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                         "grad_clip": GRAD_CLIP, "batch": BATCH_SIZE, "random_crop": True, "rotate_deg": ROTATE_DEG},
            "eval_epoch": eval_epoch, "epochs": epochs, "seed": seed, "smoke": smoke,
            "n_train_recordings": len(train_recordings), "n_val_recordings": len(val_recordings),
            "train_patients": sorted({r["patient_id"] for r in train_recordings}),
            "val_patients": sorted({r["patient_id"] for r in val_recordings})}
            
    (model_dir / "info.json").write_text(json.dumps(info, indent=1))
    (model_dir / "history.json").write_text(json.dumps(history, indent=1))
    plot_history(history, eval_epoch, model_dir / "learning_curves.png")


def predict(model_dir: Path, recordings: list[dict], device: str) -> np.ndarray:
    """
    Probabilità di Parkinson per ogni registrazione, con la normalizzazione salvata nel modello.
    """
    model_dir = Path(model_dir)
    info = json.loads((model_dir / "info.json").read_text())
    
    if info.get("input_channels") != CHANNELS or "eval_epoch" not in info:
        raise SystemExit(f"Il modello {model_dir} è stato addestrato con un altro ingresso "
                         f"(trovato {info.get('input_channels')}, atteso 'kinematic'): va riaddestrato.")
                         
    device = resolve_device(device)
    model = build_model().to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))
    
    w, owner, tasks = all_grid_windows(recordings)
    p = window_probabilities(model, to_tensor(normalize(w, tasks, info["normalization"])), device)
    
    return recording_probabilities(p, owner, len(recordings))


def plot_history(history: list[dict], eval_epoch: int, path: Path) -> None:
    """Curve di apprendimento (loss e metriche di validazione), con l'epoca del modello salvato."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train (finestre casuali)", color="#2a78d6")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="validazione (finestre fisse)", color="#eb6834")
    axes[0].set_title("Loss (BCE)", fontsize=10)
    
    for name, color in (("val_auc", "#2a78d6"), ("val_balanced_accuracy", "#eb6834"), ("val_f1", "#1baf7a")):
        axes[1].plot(epochs, [h[name] for h in history], label=name[4:].replace("_", " "), color=color)
        
    axes[1].set_title("Validazione, livello registrazione", fontsize=10)
    axes[1].set_ylim(0, 1)
    
    for name, color in (("val_sensitivity", "#eb6834"), ("val_specificity", "#1baf7a"), ("val_precision", "#6250d6")):
        axes[2].plot(epochs, [h[name] for h in history], label=name[4:], color=color)
        
    axes[2].set_title("Metriche di validazione (soglia 0.5)", fontsize=10)
    axes[2].set_ylim(0, 1)
    
    for ax in axes:
        ax.axvline(eval_epoch, color="#9a9a96", ls="--", lw=1)
        ax.set_xlabel("Epoca", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=7)
        
    fig.suptitle(f"{NAME}\n(linea verticale: epoca del modello salvato)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)